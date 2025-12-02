#!/usr/bin/env python3
# CAMBIO 009 - Ventana CREIME_RT reducida a 15 segundos para pruebas de respuesta
"""
CREIME_RT MONITOR - Sistema de Alerta Sísmica en Tiempo Real
Monitor que usa la lógica del simulador pero con datos en tiempo real de AnyShake
"""

import os
import socket
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime
import logging
from scipy.signal import butter, lfilter
import psutil
import gc
import queue
import json
import sys
import uuid
from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats

# ===== CONFIGURACIÓN GPU SEGURA PARA JETSON ORIN NANO =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir warnings TF
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Solo GPU 0

# Optimización de memoria para producción 24/7
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Limitar a 70% GPU

# Suprimir warnings adicionales
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuración de logging optimizada
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "creime_rt_monitor.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# ===== CONFIGURACIÓN DE VISUALIZACIÓN =====
VISUALIZATION_ENABLED = True
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Backend interactivo seguro
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import style
    style.use('fivethirtyeight')
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("Matplotlib no disponible - visualización desactivada")
except Exception as e:
    VISUALIZATION_ENABLED = False
    logging.warning(f"Error configurando matplotlib: {e} - visualización desactivada")

# Constantes de visualización optimizadas
DISPLAY_SECONDS = 30  # Reducido para mejor rendimiento
SAMPLING_RATE = 100
MAX_VISUALIZATION_SAMPLES = DISPLAY_SECONDS * SAMPLING_RATE

# Constantes de conversión
SENSITIVITY = 0.122
MG_TO_GALS = 0.980665
CONVERSION_FACTOR = SENSITIVITY * MG_TO_GALS

# Colores personalizados
COLOR_BLUE = '#1f77b4'    # Azul para ENZ
COLOR_ORANGE = '#ff7f0e'  # Naranja para ENE  
COLOR_GREEN = '#2ca02c'   # Verde para ENN
COLOR_RED = '#d62728'     # Rojo para CREIME_RT
COLOR_PURPLE = '#9467bd'  # Púrpura para marcadores

import threading

class RealTimeVisualizer:
    """Sistema de visualización para monitor en tiempo real"""
    
    def __init__(self, seismic_detector):
        self.detector = seismic_detector
        self.running = False
        self.fig = None
        self.axes = None
        self.animation = None
        self.visualization_enabled = VISUALIZATION_ENABLED
        
        # Buffers de datos
        self.times = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        self.data_enz = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        self.data_ene = deque(maxlen=MAX_VISUALIZATION_SAMPLES) 
        self.data_enn = deque(maxlen=MAX_VISUALIZATION_SAMPLES)
        self.creime_times = deque(maxlen=60)  # Buffer más grande para CREIME_RT
        self.creime_values = deque(maxlen=60)
        
        # Líneas de gráfico
        self.line_enz, self.line_ene, self.line_enn, self.line_creime = None, None, None, None
        self.info_text = None
        
        # Marcadores de ventana CREIME_RT sincronizados
        self.creime_markers = []
        self.processing_window_markers = []
        self.current_processing_window = None
        self.processing_window_start = None
        
        # Historial de máximos para ajuste suave
        self.max_values_history = {
            'ENZ': deque(maxlen=10),
            'ENE': deque(maxlen=10),
            'ENN': deque(maxlen=10)
        }
        
        # Estadísticas
        self.packet_count = 0
        self.start_time = time.time()
        
        self.lock = threading.Lock()
        self.setup_visualization()
    
    def setup_visualization(self):
        """Configuración de visualización para monitor"""
        if not self.visualization_enabled:
            return
            
        try:
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
            
            self.fig, self.axes = plt.subplots(4, 1, figsize=(16, 12), dpi=120)
            self.ax1, self.ax2, self.ax3, self.ax4 = self.axes
            
            self.fig.suptitle('CREIME_RT MONITOR - Datos en Tiempo Real', 
                             fontsize=16, fontweight='bold')
            
            self.line_enz, = self.ax1.plot([], [], color=COLOR_BLUE, linewidth=1.0, label='ENZ')
            self.line_ene, = self.ax2.plot([], [], color=COLOR_ORANGE, linewidth=1.0, label='ENE')
            self.line_enn, = self.ax3.plot([], [], color=COLOR_GREEN, linewidth=1.0, label='ENN')
            self.line_creime, = self.ax4.plot([], [], color=COLOR_RED, linewidth=2.0, label='CREIME_RT')
            
            components_config = [
                (self.ax1, 'Componente Vertical (ENZ)', COLOR_BLUE),
                (self.ax2, 'Componente Este-Oeste (ENE)', COLOR_ORANGE),
                (self.ax3, 'Componente Norte-Sur (ENN)', COLOR_GREEN),
                (self.ax4, 'Salida CREIME_RT (Raw Output)', COLOR_RED)
            ]
            
            for ax, title, color in components_config:
                ax.set_ylabel('Cuentas (Z-Score)', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.legend(loc='upper right')
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            self.ax4.set_xlabel('Tiempo (segundos)', fontsize=12)
            
            # Configuración especial para CREIME_RT
            self.ax4.set_ylabel('Raw Output', fontsize=12)
            self.ax4.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Umbral +2.0')
            self.ax4.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Cero')
            self.ax4.axhline(y=-4.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Ruido -4.0')
            
            self.info_text = self.ax4.text(0.02, 0.95, '', transform=self.ax4.transAxes, 
                                          fontsize=10, verticalalignment='top', color=COLOR_ORANGE,
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Texto adicional para ventana CREIME_RT
            self.creime_window_text = self.ax1.text(0.98, 0.95, '', transform=self.ax1.transAxes, 
                                                   fontsize=9, verticalalignment='top', 
                                                   horizontalalignment='right', color=COLOR_PURPLE,
                                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Configurar marcadores de ventana de procesamiento
            self.setup_processing_markers()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Configurar evento de cierre de ventana
            self.fig.canvas.mpl_connect('close_event', self.on_window_close)
            
        except Exception as e:
            logging.error(f"Error configurando visualización: {e}")
            self.visualization_enabled = False
    
    def setup_processing_markers(self):
        """Configura marcadores de ventana de procesamiento CREIME_RT"""
        if not self.visualization_enabled:
            return
            
        try:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                if ax is None:
                    continue
                # Línea vertical para marcar ventana actual de procesamiento
                window_line = ax.axvline(x=0, color=COLOR_PURPLE, linestyle='-', 
                                       linewidth=3, alpha=0.8, label='Ventana CREIME_RT (15s)')
                # Área sombreada para ventana de procesamiento
                window_fill = ax.axvspan(0, 15, color=COLOR_PURPLE, alpha=0.15)
                
                self.processing_window_markers.append({
                    'line': window_line,
                    'fill': window_fill
                })
            
            # Agregar leyenda solo en el primer subplot
            self.ax1.legend(loc='upper left', fontsize=9)
            
        except Exception as e:
            logging.error(f"Error configurando marcadores de procesamiento: {e}")
    
    def calculate_dynamic_ylimits(self, data, component_name):
        """Escalado dinámico simple basado en amplitud real de la señal"""
        if len(data) == 0:
            return -0.1, 0.1
        
        # Obtener valores máximo y mínimo reales
        data_max = np.max(data)
        data_min = np.min(data)
        
        # Calcular rango con margen del 10%
        data_range = data_max - data_min
        margin = data_range * 0.1 if data_range > 0 else 0.01
        
        # Límites dinámicos basados en amplitud real
        y_max = data_max + margin
        y_min = data_min - margin
        
        # Límite mínimo para evitar escalas muy pequeñas
        if abs(y_max - y_min) < 0.02:
            center = (y_max + y_min) / 2
            y_max = center + 0.01
            y_min = center - 0.01
        
        return y_min, y_max
    
    def update_data(self, component, data, timestamp):
        """Actualización no bloqueante de datos para visualización"""
        if not self.visualization_enabled:
            return
            
        try:
            current_time = time.time()
            
            # Usar trylock para evitar bloqueos
            if self.lock.acquire(blocking=False):
                try:
                    self.packet_count += 1
                    
                    for i, value in enumerate(data):
                        sample_time = current_time - (len(data) - i) * (1.0 / SAMPLING_RATE)
                        
                        if component == 'ENZ':
                            self.times.append(sample_time)
                            self.data_enz.append(value)
                        elif component == 'ENE':
                            self.data_ene.append(value)
                        elif component == 'ENN':
                            self.data_enn.append(value)
                finally:
                    self.lock.release()
        except:
            pass  # No bloquear por errores de visualización
    
    def mark_processing_window(self, window_start_time):
        """Marca la ventana que está siendo procesada por CREIME_RT"""
        try:
            if self.visualization_enabled:
                current_time = time.time()
                self.current_processing_window = current_time - window_start_time
                self.processing_window_start = window_start_time
        except:
            pass
    
    def update_plot(self, frame):
        """Actualización de gráficos"""
        if not self.visualization_enabled or not self.fig:
            return []
        
        with self.lock:
            times_copy = np.array(self.times)
            enz_copy = np.array(self.data_enz)
            ene_copy = np.array(self.data_ene)
            enn_copy = np.array(self.data_enn)
        
        min_len = min(len(times_copy), len(enz_copy), len(ene_copy), len(enn_copy))
        if min_len < 10:
            return []
        
        times_trim = times_copy[-min_len:]
        enz_trim = enz_copy[-min_len:]
        ene_trim = ene_copy[-min_len:]
        enn_trim = enn_copy[-min_len:]
        
        current_time_sec = time.time()
        rel_times = current_time_sec - times_trim
        
        self.line_enz.set_data(rel_times, enz_trim)
        self.line_ene.set_data(rel_times, ene_trim)
        self.line_enn.set_data(rel_times, enn_trim)
        
        xlim = (DISPLAY_SECONDS, 0)
        for ax in self.axes[:3]:  # Solo para componentes sísmicas
            ax.set_xlim(xlim)
        
        # CREIME_RT usa la misma escala de tiempo (15 segundos)
        if len(self.creime_times) > 0:
            self.ax4.set_xlim(xlim)
        else:
            self.ax4.set_xlim((15, 0))
        
        ylim_enz = self.calculate_dynamic_ylimits(enz_trim, 'ENZ')
        ylim_ene = self.calculate_dynamic_ylimits(ene_trim, 'ENE')
        ylim_enn = self.calculate_dynamic_ylimits(enn_trim, 'ENN')
        
        self.ax1.set_ylim(ylim_enz)
        self.ax2.set_ylim(ylim_ene)
        self.ax3.set_ylim(ylim_enn)
        
        # Actualizar gráfico CREIME_RT
        self.update_creime_plot()
        
        # Actualizar marcadores de ventana de procesamiento
        self.update_processing_markers(rel_times)
        
        current_time_str = datetime.now().strftime('%H:%M:%S')
        run_time = time.time() - self.start_time
        packets_per_sec = self.packet_count / run_time if run_time > 0 else 0
        
        detector_info = ""
        if hasattr(self.detector, 'detection_count'):
            detector_info = f" | Detecciones: {self.detector.detection_count}"
        if hasattr(self.detector, 'processing_count'):
            detector_info += f" | Ventanas: {self.detector.processing_count}"
        
        # Información de ventana CREIME_RT activa
        window_info = ""
        if self.current_processing_window is not None:
            window_info = f" | Ventana CREIME_RT: {self.current_processing_window:.1f}s-{max(0, self.current_processing_window-15):.1f}s"
        
        info_text = (f"Tiempo: {current_time_str} | "
                    f"Paquetes: {self.packet_count} | "
                    f"Rate: {packets_per_sec:.1f} pkt/s | "
                    f"Muestras: {min_len}{detector_info}{window_info}")
        
        self.info_text.set_text(info_text)
        
        # Actualizar texto de ventana CREIME_RT
        if hasattr(self, 'creime_window_text') and self.current_processing_window is not None:
            window_text = f"CREIME_RT procesando:\n{self.current_processing_window:.1f}s - {max(0, self.current_processing_window-15):.1f}s"
            self.creime_window_text.set_text(window_text)
        
        return [self.line_enz, self.line_ene, self.line_enn, self.line_creime]
    
    def update_creime_plot(self):
        """Actualiza el gráfico de CREIME_RT"""
        if len(self.creime_times) < 1:
            return
            
        try:
            current_time = time.time()
            creime_times_array = np.array(self.creime_times)
            creime_values_array = np.array(self.creime_values)
            
            # Convertir a tiempo relativo (segundos atrás) - igual que otros subplots
            rel_creime_times = current_time - creime_times_array
            
            self.line_creime.set_data(rel_creime_times, creime_values_array)
            
            # Límites dinámicos para CREIME_RT (sin límite máximo)
            if len(creime_values_array) > 0:
                y_min = min(-5.0, np.min(creime_values_array) - 0.5)
                y_max = max(3.0, np.max(creime_values_array) + 0.5)  # Se expande automáticamente
                self.ax4.set_ylim(y_min, y_max)
                
        except Exception as e:
            logging.debug(f"Error actualizando CREIME_RT: {e}")
    
    def add_creime_value(self, value, timestamp):
        """Añade valor CREIME_RT al gráfico"""
        if not self.visualization_enabled:
            return
            
        try:
            with self.lock:
                self.creime_times.append(timestamp)
                self.creime_values.append(value)
        except:
            pass
    
    def update_processing_markers(self, rel_times):
        """Actualiza marcadores sincronizados con ventana CREIME_RT activa"""
        if not self.processing_window_markers or len(rel_times) == 0:
            return
            
        try:
            # Mostrar ventana exacta que está procesando CREIME_RT
            if self.current_processing_window is not None:
                window_start = self.current_processing_window
                window_end = max(0, self.current_processing_window - 15)  # 15s hacia atrás
            else:
                # Ventana por defecto: últimos 15 segundos
                window_start = 15
                window_end = 0
            
            for markers in self.processing_window_markers:
                # Actualizar línea de inicio de ventana
                markers['line'].set_xdata([window_start, window_start])
                
                # Actualizar área sombreada de ventana activa
                try:
                    markers['fill'].remove()
                    ax = markers['line'].axes
                    if ax != self.ax4:  # No mostrar en gráfico CREIME_RT
                        markers['fill'] = ax.axvspan(window_start, window_end, 
                                                    color=COLOR_PURPLE, alpha=0.2,
                                                    label='Ventana CREIME_RT Activa')
                except:
                    pass
                    
        except Exception as e:
            logging.debug(f"Error actualizando marcadores: {e}")
    
    def start_visualization(self):
        """Inicia visualización en hilo separado"""
        if not self.visualization_enabled or not self.fig:
            return
            
        try:
            self.running = True
            # Crear animación en hilo separado para no bloquear
            self.viz_thread = threading.Thread(
                target=self._run_visualization,
                name="Visualizer",
                daemon=True
            )
            self.viz_thread.start()

        except Exception as e:
            logging.error(f"Error iniciando visualización: {e}")
            self.visualization_enabled = False
    
    def _run_visualization(self):
        """Ejecuta visualización en hilo separado"""
        try:
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=1000, blit=False, cache_frame_data=False
            )
        except Exception as e:
            logging.warning(f"Error en hilo de visualización: {e}")
    
    def on_window_close(self, event):
        """Maneja el cierre de la ventana del visualizador"""
        logging.info("Ventana cerrada - Deteniendo sistema")
        self.detector.stop_monitor()
    
    def stop_visualization(self):
        """Detiene el sistema de visualización"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        logging.info("Visualizador detenido")

class UltraFastBuffer:
    """Buffer de latencia mínima - ventana deslizante continua"""
    
    def __init__(self, window_size=3000, sampling_rate=100, update_interval=1.0):
        self.window_size = int(window_size)
        self.sampling_rate = int(sampling_rate)
        self.update_interval = update_interval  # 1000ms sincronizado con AnyShake
        
        # Buffer circular exacto para ventana de 30s
        self.buffers = {
            'ENZ': deque(maxlen=self.window_size),
            'ENE': deque(maxlen=self.window_size),
            'ENN': deque(maxlen=self.window_size)
        }
        
        self.lock = threading.Lock()
        self.window_count = 0
        self.last_window_time = 0
        self.ready = False
        self.min_ready_samples = self.window_size  # Requiere ventana completa
        self.new_data_event = threading.Event()
        
        self.performance_stats = {
            'windows_generated': 0,
            'data_points_received': 0,
            'last_update': time.time()
        }
    
    def add_data(self, component, data, timestamp):
        """Añade datos del monitor en tiempo real"""
        with self.lock:
            if component in self.buffers:
                self.buffers[component].extend(data)
                self.performance_stats['data_points_received'] += len(data)
                
                min_samples = min(len(buf) for buf in self.buffers.values())
                self.ready = min_samples >= self.min_ready_samples
                
                if data:
                    self.new_data_event.set()
    
    def wait_for_new_data(self, timeout=0.5):
        """Espera por nuevos datos"""
        return self.new_data_event.wait(timeout)
    
    def reset_data_event(self):
        """Resetea el evento de nuevos datos"""
        self.new_data_event.clear()
    
    def get_latest_window(self):
        """Ventana deslizante ultra-rápida para latencia <1.5s"""
        with self.lock:
            current_time = time.time()
            
            # CRÍTICO: Procesar cada 500ms para latencia mínima
            if current_time - self.last_window_time < 0.5:
                return None
                
            if not self.ready:
                return None
            
            window_data = []
            for component in ['ENZ', 'ENE', 'ENN']:
                buf = self.buffers[component]
                
                # Usar directamente el buffer circular (siempre los últimos 15s)
                if len(buf) == self.window_size:
                    component_data = list(buf)  # Los últimos 1500 datos
                else:
                    # Rellenar solo si no tenemos ventana completa aún
                    padding_needed = self.window_size - len(buf)
                    component_data = [0.0] * padding_needed + list(buf)
                
                window_data.append(component_data)
            
            try:
                # Usar directamente 1500 muestras (15 segundos) para CREIME_RT
                window_3d = np.stack([window_data[0], window_data[1], window_data[2]], axis=1)
                window_3d = np.expand_dims(window_3d, axis=0).astype(np.float32)
                
                self.window_count += 1
                self.performance_stats['windows_generated'] += 1
                self.last_window_time = current_time
                
                return window_3d
                
            except Exception as e:
                logging.error(f"Error creando ventana 3D: {e}")
                return None
    
    def get_buffer_status(self):
        """Estado del buffer"""
        with self.lock:
            status = {}
            for comp, buf in self.buffers.items():
                status[comp] = {
                    'samples': len(buf),
                    'percent': (len(buf) / self.window_size) * 100,
                    'seconds': len(buf) / self.sampling_rate
                }
            
            status['ready'] = self.ready
            status['min_required'] = f"{self.min_ready_samples} muestras ({self.min_ready_samples/self.sampling_rate:.1f}s)"
            
            return status

class OptimizedSeismicFilter:
    """Filtro pasa-bandas + Z-Score en tiempo real (latencia mínima)"""
    
    def __init__(self, fs=100, lowcut=1.0, highcut=45.0, order=4):
        self.fs = fs
        
        # Filtro pasa-bandas IIR Butterworth (solución del documento)
        from scipy.signal import butter, sosfilt_zi
        
        try:
            # Diseño del filtro en formato SOS (super estable para tiempo real)
            self.sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
            
            # Diagnóstico detallado
            logging.info(f"SOS shape: {self.sos.shape}, dtype: {self.sos.dtype}")
            logging.info(f"SOS primeras secciones: {self.sos[:2] if len(self.sos) > 0 else 'vacío'}")
            
            # Validar formato SOS antes de usar sosfilt_zi
            if self.sos.ndim != 2 or self.sos.shape[1] != 6:
                raise ValueError(f"SOS formato incorrecto: {self.sos.shape}, esperado (n, 6)")
            
            # Estado inicial para cada componente - sección por sección
            def create_filter_state():
                states = []
                for i in range(self.sos.shape[0]):
                    section = self.sos[i]  # Extraer sección individual
                    zi = sosfilt_zi(section.reshape(1, 6))  # Asegurar forma (1, 6)
                    states.append(zi)
                return states
            
            self.filter_states = {
                'ENZ': create_filter_state(),
                'ENE': create_filter_state(),
                'ENN': create_filter_state()
            }
            
            logging.info(f"Filtro inicializado: {self.sos.shape[0]} secciones, {lowcut}-{highcut}Hz")
            
        except Exception as e:
            logging.error(f"Error inicializando filtro: {e}")
            logging.error(f"Parámetros: fs={fs}, lowcut={lowcut}, highcut={highcut}, order={order}")
            # Fallback: sin filtrado
            self.sos = None
            self.filter_states = {'ENZ': None, 'ENE': None, 'ENN': None}
        
        # Sin normalización Z-Score - usar datos filtrados directamente
        self.current_component = None
    

    

    
    def set_component(self, component):
        """Establece el componente actual para normalización"""
        self.current_component = component
    
    def _apply_bandpass_filter(self, data, component):
        """Aplica filtro pasa-bandas IIR - implementación del documento"""
        if self.sos is None or component not in self.filter_states or self.filter_states[component] is None:
            return data  # Sin filtrado si hay error o componente desconocido
        
        try:
            from scipy.signal import sosfilt
            
            # Aplicar filtro sección por sección
            filtered = np.array(data, dtype=np.float64).copy()
            zi_states = self.filter_states[component]
            
            for i in range(self.sos.shape[0]):
                section = self.sos[i].reshape(1, 6)  # Asegurar forma correcta
                filtered, zi_states[i] = sosfilt(section, filtered, zi=zi_states[i])
            
            # Actualizar estado para próximo paquete
            self.filter_states[component] = zi_states
            
            return filtered.tolist()
            
        except Exception as e:
            logging.warning(f"Error en filtro para {component}: {e}")
            return data  # Retornar datos sin filtrar si hay error
    
    def apply_filter(self, data, component=None):
        """Pipeline sin normalización: Solo filtro + escala fija"""
        if not data:
            return []
        
        # PASO 1: Filtro pasa-bandas 1-40Hz
        filtered_data = self._apply_bandpass_filter(data, component)
        
        # PASO 2: Conversión a Gals como AnyShake-data-stream.py
        gals_data = [x * CONVERSION_FACTOR for x in filtered_data]  # 0.119 Gals
        
        return gals_data

class UltraFastProcessingPipeline:
    """Pipeline de procesamiento para monitor"""
    
    def __init__(self, model_path, num_workers=1):
        self.model_path = model_path
        self.num_workers = num_workers
        self.processing_queue = queue.Queue(maxsize=3)  # Buffer optimizado para latencia
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        
        self.start_workers()
    
    def start_workers(self):
        """Inicia workers de procesamiento"""
        self.running = True
        self.workers_ready = threading.Event()  # Evento para sincronizar inicialización
        self.worker_initialized = False
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self.processing_worker,
                name=f"CREIME_Monitor_Worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def processing_worker(self):
        """Worker para CREIME_RT en monitor"""
        try:
            # Configurar TensorFlow para uso eficiente de memoria
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Habilitar crecimiento de memoria dinámico
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    # Limitar memoria GPU a 1GB para CREIME_RT
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                    )
                except RuntimeError as e:
                    logging.warning(f"Error configurando GPU: {e}")
            
            from saipy.models.creime import CREIME_RT
            # Validar directorio del modelo para prevenir CWE-94
            if not os.path.isdir(self.model_path):
                raise ValueError(f"Directorio de modelo inválido: {self.model_path}")
            
            # CREIME_RT espera el directorio donde están los archivos del modelo
            model = CREIME_RT(self.model_path)
            
            # Forzar garbage collection después de cargar modelo
            import gc
            gc.collect()
            
            # Señalar que el worker está listo
            self.worker_initialized = True
            self.workers_ready.set()
            
        except Exception as e:
            logging.error(f"Worker monitor no pudo cargar modelo: {e}")
            self.worker_initialized = False
            self.workers_ready.set()  # Señalar fallo también
            return
        
        while self.running:
            try:
                window_data, processing_id = self.processing_queue.get(timeout=1.0)
                
                if window_data is None:
                    break
                
                start_time = time.time()
                
                try:
                    y_pred, predictions = model.predict(window_data)
                    
                    # Usar el valor máximo del vector completo (6000 muestras)
                    if y_pred is not None and len(y_pred.shape) > 1:
                        raw_output = float(np.max(y_pred[0]))  # Máximo del vector completo
                    else:
                        raw_output = -4.0
                    
                    # UMBRAL ALTO: +2.0
                    if raw_output > 2.0:
                        detection = 1
                        magnitude = raw_output if raw_output > 0 else 0.0
                        # DIAGNÓSTICO: Log inmediato de detección
                        logging.critical(f"DETECCIÓN INMEDIATA: Raw={raw_output:.3f} > +2.0 en ventana {processing_id}")
                    else:
                        detection = 0
                        magnitude = None
                    
                    result = (detection, magnitude, raw_output)
                    
                except (IndexError, ValueError, TypeError) as e:
                    logging.warning(f"Error en predicción monitor: {e}")
                    result = (0, None, -4.0)
                except Exception as e:
                    logging.error(f"Error inesperado en monitor: {e}")
                    result = (0, None, -4.0)
                
                processing_time = time.time() - start_time
                
                self.result_queue.put({
                    'result': result,
                    'processing_id': processing_id,
                    'processing_time': processing_time,
                    'window_data': window_data
                })
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error en worker monitor: {e}")
                try:
                    self.processing_queue.task_done()
                except:
                    pass
                continue
    
    def submit_window(self, window_data, processing_id):
        """Envía ventana para procesamiento"""
        try:
            self.processing_queue.put((window_data, processing_id), timeout=0.2)  # Timeout agresivo
            return True
        except queue.Full:
            # CRÍTICO: Descartar ventana antigua para mantener latencia
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.put((window_data, processing_id), timeout=0.1)
                return True
            except:
                return False
    
    def get_result(self, timeout=0.5):
        """Obtiene resultado"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_worker_ready(self, timeout=30):
        """Verifica si el worker CREIME_RT está listo"""
        if hasattr(self, 'workers_ready') and self.workers_ready.wait(timeout):
            return getattr(self, 'worker_initialized', False)
        return False
    
    def stop_workers(self):
        """Detiene workers"""
        self.running = False
        for _ in range(self.num_workers):
            try:
                self.processing_queue.put((None, None), timeout=0.5)
            except queue.Full:
                break
        
        for worker in self.workers:
            worker.join(timeout=2.0)

class RealTimeMonitor:
    """
    Monitor principal que usa datos en tiempo real de AnyShake
    """
    
    def __init__(self, model_path, host='localhost', port=30000, sampling_rate=100):
        # Validación de seguridad para prevenir CWE-94
        if not isinstance(model_path, str) or not os.path.isdir(model_path):
            raise ValueError("model_path debe ser un directorio válido")
        if not isinstance(host, str) or len(host) > 253:
            raise ValueError("host inválido")
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError("port debe estar entre 1-65535")
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError("sampling_rate debe ser positivo")
            
        self.model_path = model_path
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        
        # Parámetros sincronizados con AnyShake (se ajustarán dinámicamente)
        self.window_size = 15 * sampling_rate  # 1500 muestras - 15 SEGUNDOS
        self.anyshake_packet_interval = 1.0  # Inicial - se actualizará según modo
        self.latency_target = self.anyshake_packet_interval  # Sincronización dinámica
        self.detection_threshold = 2.0  # Umbral alto para eventos significativos
        self.noise_baseline = -4.0
        self.high_noise_threshold = -1.80
        self.magnitude_threshold = 0.0  # Umbral original para magnitud
        self.consecutive_windows = 1  # Confirmación inmediata para latencia mínima
        
        # Componentes del sistema
        self.buffer = UltraFastBuffer(
            window_size=self.window_size,
            sampling_rate=sampling_rate,
            update_interval=self.anyshake_packet_interval  # Sincronizado con AnyShake
        )
        
        self.seismic_filter = OptimizedSeismicFilter(fs=sampling_rate, lowcut=1.0, highcut=40.0, order=4)
        self.processing_pipeline = UltraFastProcessingPipeline(model_path, num_workers=1)
        self.visualizer = RealTimeVisualizer(self)
        
        # Estado del sistema
        self.running = False
        self.socket = None
        self.data_buffer = b''
        
        # Estadísticas
        self.detection_count = 0
        self.anomaly_count = 0
        self.confirmed_earthquake_count = 0
        self.last_detection_time = None
        self.packet_count = 0
        self.start_time = None
        self.processing_count = 0
        self.last_processing_time = 0
        
        # Buffer para confirmación de sismos (3 anomalías consecutivas)
        self.anomaly_buffer = deque(maxlen=3)
        self.consecutive_anomalies = 0
        
        # Configuración de estación
        self.station_id = "CREIME_RT_MONITOR"
        
        # Rastreo de eventos y timestamps
        self.detected_events = []
        self.monitor_start_time = None
        
        # Rastreo de tiempos de detección
        self.first_detection_time = None  # Primer evento detectado
        self.first_confirmation_time = None  # Primera confirmación de sismo
        
        # Rastreo de valores CREIME_RT para diagnóstico
        self.creime_values = []
        self.creime_timestamps = []  # Timestamps correspondientes
        
        # Buffer para MiniSEED (15 segundos antes del evento)
        self.miniseed_buffer = {
            'ENZ': deque(maxlen=1500),  # 15 segundos a 100Hz
            'ENE': deque(maxlen=1500),
            'ENN': deque(maxlen=1500)
        }
        self.miniseed_timestamps = deque(maxlen=1500)
        
        # Hilos
        self.data_thread = None
        self.processing_thread = None
        
        # Control de parada
        self.stop_called = False
        
        logging.info(f"Monitor CREIME_RT configurado - {host}:{port}")
    
    def enable_anyshake_realtime(self):
        """Activa modo tiempo real en AnyShake - protocolo correcto"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            
            # Enviar comando REALTIME según ANYSHAKE RULES
            self.socket.sendall(b"AT+REALTIME=1\r\n")
            time.sleep(1)
            
            # Configurar para recepción no bloqueante
            self.socket.setblocking(False)
            
            logging.info("Modo REALTIME activado - Conexión establecida")
            return True
        except Exception as e:
            logging.warning(f"Error activando tiempo real: {e}")
            return False
    
    def connect_to_anyshake(self):
        """Conexión con AnyShake Observer"""
        # Activar modo tiempo real
        self.enable_anyshake_realtime()
        
        # Configuración estable para modo REALTIME
        self.anyshake_packet_interval = 1.0  # Mantener 1 segundo inicial
        self.latency_target = 1.0
        self.buffer.update_interval = 1.0
        logging.info("Modo REALTIME activado - Esperando datos cada 1s")
        
        time.sleep(2)  # Esperar activación
        logging.info("Sistema configurado en modo estable - Buffer 1s")
        
        if self.socket is None:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                self.socket.connect((self.host, self.port))
            except Exception as e:
                logging.error(f"Error conectando: {e}")
                return False
        
        logging.info(f"Conexión establecida: {self.host}:{self.port}")
        return True
    
    def parse_observer_packet(self, data_line):
        """Parser correcto para paquetes AnyShake - copiado de AnyShake-data-stream"""
        try:
            data_line = data_line.strip()
            
            # Validar formato básico
            if not data_line.startswith('$'):
                return None
            
            # Buscar el checksum
            if '*' not in data_line:
                return None
                
            main_part, checksum_part = data_line.split('*', 1)
            checksum = checksum_part.replace('\r', '')
            
            # Dividir por comas
            parts = main_part.split(',')
            if len(parts) < 8:
                return None
            
            # Extraer campos según especificación
            sequence = parts[0]          # $N
            manufacturer = parts[1]      # AS
            device_type = parts[2]       # SHAKE  
            station_id = parts[3]        # ID
            component = parts[4]         # COMP (ENZ, ENE, ENN)
            timestamp = int(parts[5])    # TIMESTAMP (ms desde epoch)
            sample_rate = int(parts[6])  # FS (Hz)
            
            # Validar sample rate
            if sample_rate != self.sampling_rate:
                return None
            
            # Extraer datos de aceleración (valores enteros)
            data_values = []
            for i in range(7, len(parts)):
                part = parts[i].strip()
                if part and not part.startswith('*'):  # Evitar el checksum
                    try:
                        data_values.append(int(part))
                    except ValueError:
                        continue
            
            if not data_values:
                return None
                
            # Pipeline completo: Filtro pasa-bandas → Z-Score Welford
            processed_data = self.seismic_filter.apply_filter(data_values, component)
                
            return {
                'component': component,
                'timestamp': timestamp,
                'sampling_rate': sample_rate,
                'data': processed_data,
                'sequence': sequence,
                'checksum': checksum
            }
            
        except Exception as e:
            logging.debug(f"Error parseando paquete: {e}")
            return None
    
    def ultra_fast_processing(self):
        """Procesamiento ultra-rápido para alerta sísmica temprana"""
        current_time = time.time()
        
        # CRÍTICO: Procesar cada 100ms para detección inmediata
        min_interval = 0.1  # Máxima frecuencia para detección temprana
        if current_time - self.last_processing_time < min_interval:
            return None
        
        window_data = self.buffer.get_latest_window()
        if window_data is None:
            return None
        
        processing_id = self.processing_count
        if self.processing_pipeline.submit_window(window_data, processing_id):
            self.processing_count += 1
            self.last_processing_time = current_time
            
            result_data = self.processing_pipeline.get_result(timeout=0.3)  # Timeout agresivo
            
            if result_data:
                processing_time = time.time() - current_time
                
                return {
                    'timestamp': datetime.now(),
                    'detection': result_data['result'][0],
                    'magnitude': result_data['result'][1],
                    'confidence': self._calculate_confidence(result_data),
                    'processing_id': processing_id,
                    'processing_time': processing_time,
                    'window_data': window_data,
                    'window_start_time': current_time - 30  # Inicio de ventana de 30s
                }
        
        return None
    
    def _calculate_confidence(self, result_data):
        """Interpreta salida cruda de CREIME_RT"""
        try:
            if len(result_data['result']) >= 3:
                raw_output = float(result_data['result'][2])
            else:
                raw_output = self.noise_baseline
            
            # Log normal de valores CREIME_RT
            current_time = datetime.now()
            logging.info(f"[{current_time}] CREIME_RT Raw Output: {raw_output:.3f}")
            
            return raw_output
            
        except (IndexError, TypeError, ValueError) as e:
            logging.warning(f"Error extrayendo salida CREIME_RT: {e}")
            current_time = datetime.now()
            logging.info(f"[{current_time}] CREIME_RT Raw Output: {self.noise_baseline:.2f} (error)")
            return self.noise_baseline
    
    def evaluate_detection(self, result):
        """Sistema de clasificación de tres niveles"""
        if not result:
            return False
            
        confidence = result['confidence']
        
        # 1. Anomalía detectada: supera umbral +2.0
        if confidence > self.detection_threshold:
            self.anomaly_buffer.append(True)
            self.consecutive_anomalies += 1
            self.anomaly_count += 1
            
            logging.warning(f"🟡 ANOMALÍA DETECTADA: Confianza {confidence:.2f} > +2.0")
            
            # 2. Sismo confirmado: 3 anomalías consecutivas
            if self.consecutive_anomalies >= 3:
                self.confirmed_earthquake_count += 1
                logging.critical(f"🚨 SISMO CONFIRMADO 🚨 - 3 anomalías consecutivas")
                
                return {
                    'type': 'confirmed_earthquake',
                    'consecutive_anomalies': self.consecutive_anomalies,
                    'is_seismic': True
                }
            else:
                return {
                    'type': 'anomaly_detected', 
                    'consecutive_anomalies': self.consecutive_anomalies,
                    'is_seismic': False
                }
        else:
            # 3. Ruido: resetear contador de anomalías consecutivas
            if self.consecutive_anomalies > 0:
                self.consecutive_anomalies = 0
                self.anomaly_buffer.clear()
            
            return False
    

    
    def _is_seismic_event(self, result):
        """Clasificación original CREIME_RT"""
        return result['confidence'] > self.magnitude_threshold
    
    def trigger_alert(self, detection_result, detection_info):
        """Activa alerta según sistema de tres niveles"""
        self.detection_count += 1
        self.last_detection_time = detection_result['timestamp']
        
        # Calcular tiempo real del evento
        event_time = detection_result['timestamp']
        
        # Registrar primera detección
        if self.first_detection_time is None:
            self.first_detection_time = event_time
        
        if detection_info['type'] == 'confirmed_earthquake':
            # SISMO CONFIRMADO: Solo aquí se generan archivos JSON y MiniSEED
            if self.first_confirmation_time is None:
                self.first_confirmation_time = event_time
            
            # Usar salida cruda de CREIME_RT como magnitud directamente
            raw_magnitude = detection_result['confidence']
            
            logging.critical(f"🚨 SISMO CONFIRMADO 🚨 - Magnitud: {raw_magnitude:.2f}")
            
            # Registrar evento confirmado
            self.detected_events.append({
                'type': 'confirmed_earthquake',
                'event_time': event_time,
                'confidence': detection_result['confidence'],
                'magnitude': raw_magnitude,
                'processing_id': detection_result['processing_id'],
                'consecutive_anomalies': detection_info['consecutive_anomalies']
            })
            
            # SOLO generar archivos para sismos confirmados
            self.save_event_data(detection_result, raw_magnitude)
            
        elif detection_info['type'] == 'anomaly_detected':
            # ANOMALÍA DETECTADA: Solo log, sin archivos
            mag_display = f"{detection_result['magnitude']:.1f}" if detection_result['magnitude'] is not None else "N/A"
            
            # Registrar anomalía (sin archivos)
            self.detected_events.append({
                'type': 'anomaly',
                'event_time': event_time,
                'confidence': detection_result['confidence'],
                'magnitude': detection_result['magnitude'],
                'processing_id': detection_result['processing_id'],
                'consecutive_anomalies': detection_info['consecutive_anomalies']
            })
    
    def save_event_data(self, detection_result, raw_magnitude):
        """Guarda datos del evento detectado en monitor"""
        try:
            events_dir = "events_monitor"
            if not os.path.exists(events_dir):
                os.makedirs(events_dir)
            
            event_id = str(uuid.uuid4())[:8]
            timestamp_str = detection_result['timestamp'].strftime('%Y%m%d_%H%M%S')
            
            # 1. Guardar JSON inmediatamente
            json_data = {
                "station_id": self.station_id,
                "event_id": event_id,
                "timestamp": detection_result['timestamp'].isoformat(),
                "raw_output": detection_result['confidence'],
                "magnitude": raw_magnitude
            }
            
            json_filename = os.path.join(events_dir, f"monitor_event_{timestamp_str}.json")
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            logging.info(f"Evento monitor guardado: {json_filename}")
            
            # 2. Programar MiniSEED para 60 segundos después
            miniseed_timer = threading.Timer(60.0, self._save_miniseed, 
                                           args=[event_id, timestamp_str, detection_result['timestamp']])
            miniseed_timer.daemon = True
            miniseed_timer.start()
            logging.info(f"MiniSEED programado para 60s: monitor_event_{timestamp_str}.mseed")
            
        except Exception as e:
            logging.error(f"Error guardando evento monitor: {e}")
    
    def _save_miniseed(self, event_id, timestamp_str, event_time):
        """Guarda archivo MiniSEED con 1 minuto de datos (15s antes del evento)"""
        try:
            events_dir = "events_monitor"
            
            # Crear stream ObsPy
            stream = Stream()
            
            # Obtener datos de los últimos 60 segundos (6000 muestras)
            for i, component in enumerate(['ENZ', 'ENE', 'ENN']):
                channel_code = ['HHZ', 'HHE', 'HHN'][i]
                
                # Extraer últimas 6000 muestras (60 segundos)
                buffer_data = list(self.miniseed_buffer[component])[-6000:] if len(self.miniseed_buffer[component]) >= 6000 else list(self.miniseed_buffer[component])
                
                if len(buffer_data) < 6000:
                    # Rellenar con ceros si no hay suficientes datos
                    buffer_data = [0.0] * (6000 - len(buffer_data)) + buffer_data
                
                # Validar datos para MiniSEED
                if not buffer_data or all(x == 0 for x in buffer_data):
                    logging.warning(f"Datos vacíos para {component}, omitiendo")
                    continue
                
                # Crear trace con validación
                stats = Stats()
                stats.network = "SK"
                stats.station = "MONITOR"
                stats.location = "00"
                stats.channel = channel_code
                stats.sampling_rate = 100.0
                stats.starttime = UTCDateTime(event_time) - 15
                stats.npts = len(buffer_data)
                
                # Convertir a enteros para MiniSEED (evita errores de empaquetado)
                int_data = np.array(buffer_data, dtype=np.int32)
                trace = Trace(data=int_data, header=stats)
                stream.append(trace)
            
            # Guardar MiniSEED
            mseed_filename = os.path.join(events_dir, f"monitor_event_{timestamp_str}.mseed")
            stream.write(mseed_filename, format='MSEED')
            logging.info(f"MiniSEED guardado: {mseed_filename}")
            
        except Exception as e:
            logging.error(f"Error guardando MiniSEED: {e}")
    
    def processing_loop(self):
        """Bucle de procesamiento con estabilidad 24/7"""
        self.last_processing_time = time.time()
        cycle_count = 0
        
        while self.running:
            try:
                if self.buffer.wait_for_new_data(timeout=0.6):  # Timeout más agresivo
                    self.buffer.reset_data_event()
                    
                    result = self.ultra_fast_processing()
                    
                    if result:
                        # Marcar ventana sincronizada en visualizador
                        if hasattr(result, 'window_start_time'):
                            self.visualizer.mark_processing_window(result['window_start_time'])
                        
                        # Capturar valor y timestamp para estadísticas
                        self.creime_values.append(result['confidence'])
                        self.creime_timestamps.append(result['timestamp'])
                        
                        # Actualizar visualizador CREIME_RT
                        self.visualizer.add_creime_value(result['confidence'], time.time())
                        
                        detection_info = self.evaluate_detection(result)
                        if detection_info:
                            self.trigger_alert(result, detection_info)
                
                # Limpieza periódica de memoria (frecuencia según modo)
                cycle_count += 1
                cleanup_interval = 900 if getattr(self, 'production_mode', False) else 3600  # 15 min prod, 1 hora normal
                if cycle_count % cleanup_interval == 0:
                    try:
                        self._memory_cleanup()
                    except:
                        pass  # No bloquear por errores de limpieza
                
                time.sleep(0.05)  # Ciclo más rápido para tiempo real
                
            except Exception as e:
                logging.error(f"Error en bucle procesamiento monitor: {e}")
                self._handle_processing_error()
                time.sleep(0.1)
    
    def receive_data_loop(self):
        """Bucle de recepción de datos AnyShake"""
        self.running = True
        self.data_buffer = b''
        
        while self.running:
            try:
                # Usar select para evitar errores de socket no bloqueante
                import select
                ready = select.select([self.socket], [], [], 0.05)  # Timeout ultra-corto
                if ready[0]:
                    data = self.socket.recv(4096)
                    if not data:
                        logging.warning("Conexión cerrada - Reconectando...")
                        if not self.connect_to_anyshake():
                            time.sleep(3)
                            continue
                        else:
                            continue
                else:
                    # No hay datos disponibles, continuar
                    time.sleep(0.005)  # Polling ultra-frecuente
                    continue
                
                self.data_buffer += data
                
                while b'\n' in self.data_buffer or b'\r' in self.data_buffer:
                    if b'\n' in self.data_buffer:
                        packet, self.data_buffer = self.data_buffer.split(b'\n', 1)
                    else:
                        packet, self.data_buffer = self.data_buffer.split(b'\r', 1)
                    
                    try:
                        packet_str = packet.decode('ascii', errors='ignore').strip()
                        if packet_str:
                            parsed_data = self.parse_observer_packet(packet_str)
                            if parsed_data:
                                current_time = time.time()
                                
                                self.buffer.add_data(
                                    parsed_data['component'],
                                    parsed_data['data'],
                                    current_time
                                )
                                
                                # Actualizar buffer MiniSEED
                                if parsed_data['component'] in self.miniseed_buffer:
                                    self.miniseed_buffer[parsed_data['component']].extend(parsed_data['data'])
                                    for _ in parsed_data['data']:
                                        self.miniseed_timestamps.append(current_time)
                                
                                # Actualizar visualizador de forma no bloqueante
                                try:
                                    self.visualizer.update_data(
                                        parsed_data['component'],
                                        parsed_data['data'],
                                        current_time
                                    )
                                except:
                                    pass  # No bloquear por errores de visualización
                                
                                self.packet_count += 1
                                
                                # Monitorear y ajustar dinámicamente el modo de AnyShake
                                if self.packet_count % 15 == 0:  # Cada 15 paquetes
                                    elapsed = current_time - self.start_time
                                    packet_rate = self.packet_count / elapsed if elapsed > 0 else 0
                                    expected_rate = 3.0 / self.anyshake_packet_interval  # 3 componentes
                                    
                                    # Detectar modo de operación
                                    if packet_rate > 8:
                                        mode_status = "TIEMPO REAL"
                                    elif packet_rate > 2.5:
                                        mode_status = "NORMAL"
                                    else:
                                        mode_status = "LENTO"
                                    
                                    if self.packet_count == 15:
                                        logging.info(f"Tasa: {packet_rate:.1f} pkt/s - {mode_status}")
                                        if packet_rate > 2.5:
                                            logging.info("AT+REALTIME=1 exitoso - Sistema estable")
                                
                    except Exception as e:
                        continue
                        
            except socket.timeout:
                continue
            except (BlockingIOError, socket.error) as e:
                # Errores normales de socket no bloqueante - no logear
                time.sleep(0.01)  # Reintentos ultra-rápidos
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error en recepción: {e}")
                    time.sleep(3)
                else:
                    break
    
    def plot_creime_output_timeline(self):
        """Genera gráfico de raw output CREIME_RT vs tiempo"""
        if not self.creime_values or not VISUALIZATION_ENABLED:
            return
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Convertir timestamps a datetime
            if self.creime_timestamps:
                times = [ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00')) 
                        for ts in self.creime_timestamps]
            else:
                times = list(range(len(self.creime_values)))
            
            # Crear gráfico
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Plotear raw output
            ax.plot(times, self.creime_values, 'b-', linewidth=1.0, alpha=0.8, label='CREIME_RT Raw Output')
            
            # Líneas de umbrales
            ax.axhline(y=self.detection_threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Umbral Detección ({self.detection_threshold})')
            ax.axhline(y=self.noise_baseline, color='gray', linestyle='--', linewidth=1, 
                      label=f'Línea Base Ruido ({self.noise_baseline})')
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1, 
                      label='Cero (Magnitud Positiva)')
            ax.axhline(y=self.magnitude_threshold, color='orange', linestyle='--', linewidth=1, 
                      label=f'Umbral Magnitud ({self.magnitude_threshold})')
            
            # Configuración del gráfico
            ax.set_xlabel('Tiempo', fontsize=12)
            ax.set_ylabel('CREIME_RT Raw Output', fontsize=12)
            ax.set_title('Evolución Temporal CREIME_RT - Monitor en Tiempo Real', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Formatear eje X si son timestamps reales
            if self.creime_timestamps and isinstance(self.creime_timestamps[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Estadísticas en el gráfico
            stats_text = (
                f'Valores: {len(self.creime_values)}\n'
                f'Mínimo: {min(self.creime_values):.3f}\n'
                f'Máximo: {max(self.creime_values):.3f}\n'
                f'Promedio: {sum(self.creime_values)/len(self.creime_values):.3f}\n'
                f'Detecciones: {sum(1 for v in self.creime_values if v > self.detection_threshold)}'
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            plt.show()
            
            logging.info("Gráfico de raw output generado")
            
        except Exception as e:
            logging.error(f"Error generando gráfico: {e}")
    
    def start_monitor(self):
        """Inicia el monitor completo"""
        if not self.connect_to_anyshake():
            return False
        
        self.start_time = time.time()
        self.monitor_start_time = datetime.now()
        
        logging.info(f"Monitor iniciado - {self.host}:{self.port}")
        
        self.running = True
        
        # Hilo de recepción de datos
        self.data_thread = threading.Thread(
            target=self.receive_data_loop,
            name="AnyShakeReceiver",
            daemon=True
        )
        self.data_thread.start()
        
        worker_ready = self.processing_pipeline.is_worker_ready(timeout=60)
        
        if worker_ready:
            logging.info("CREIME_RT listo")
            
            # Hilo de procesamiento
            self.processing_thread = threading.Thread(
                target=self.processing_loop,
                name="MonitorProcessor", 
                daemon=True
            )
            self.processing_thread.start()
            
            if VISUALIZATION_ENABLED:
                self.visualizer.start_visualization()
        else:
            logging.error("Worker CREIME_RT no se inicializó correctamente")
            self.stop_monitor()
            return False
        
        # Esperar inicialización
        buffer_ready = False
        startup_time = time.time()
        while not buffer_ready and self.running and (time.time() - startup_time < 10):
            time.sleep(0.1)
            status = self.buffer.get_buffer_status()
            current_samples = status['ENZ']['samples']
            buffer_ready = current_samples >= 50

        logging.info("Monitor operativo")
        
        return True
    
    def _memory_cleanup(self):
        """Limpieza periódica de memoria optimizada para producción 24/7"""
        try:
            # Límites según modo
            if getattr(self, 'production_mode', False):
                max_values = 1800  # 30 min en producción
                keep_values = 900   # Mantener 15 min
                max_events = 50
                keep_events = 25
            else:
                max_values = 10000  # Modo normal
                keep_values = 5000
                max_events = 100
                keep_events = 50
            
            # Limpiar buffers de estadísticas
            if len(self.creime_values) > max_values:
                self.creime_values = self.creime_values[-keep_values:]
                self.creime_timestamps = self.creime_timestamps[-keep_values:]
            
            # Limpiar eventos antiguos
            if len(self.detected_events) > max_events:
                self.detected_events = self.detected_events[-keep_events:]
            
            # Limpiar buffer de visualización en modo producción
            if getattr(self, 'production_mode', False) and hasattr(self.visualizer, 'times'):
                max_vis_samples = 30 * self.sampling_rate  # 30 segundos en producción
                if len(self.visualizer.times) > max_vis_samples:
                    with self.visualizer.lock:
                        self.visualizer.times = deque(list(self.visualizer.times)[-max_vis_samples:], maxlen=max_vis_samples)
                        self.visualizer.data_enz = deque(list(self.visualizer.data_enz)[-max_vis_samples:], maxlen=max_vis_samples)
                        self.visualizer.data_ene = deque(list(self.visualizer.data_ene)[-max_vis_samples:], maxlen=max_vis_samples)
                        self.visualizer.data_enn = deque(list(self.visualizer.data_enn)[-max_vis_samples:], maxlen=max_vis_samples)
            
            # Garbage collection agresivo
            import gc
            gc.collect(0)
            gc.collect(1) 
            gc.collect(2)
            
            # Log de uso de memoria
            memory_percent = psutil.virtual_memory().percent
            mode = "PRODUCCIÓN" if getattr(self, 'production_mode', False) else "NORMAL"
            logging.info(f"Limpieza {mode} completada - RAM: {memory_percent:.1f}%")
            
        except Exception as e:
            logging.warning(f"Error en limpieza de memoria: {e}")
    
    def _handle_processing_error(self):
        """Manejo de errores de procesamiento para recuperación automática"""
        try:
            # Intentar reinicializar pipeline si hay errores consecutivos
            logging.warning("Intentando recuperación automática del pipeline")
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error en recuperación automática: {e}")
    
    def stop_monitor(self):
        """Detiene el monitor con limpieza completa"""
        if self.stop_called:
            return  # Evitar múltiples llamadas
        
        self.stop_called = True
        logging.info("Iniciando parada segura del monitor...")
        self.running = False
        
        # Parada ordenada de componentes
        try:
            self.processing_pipeline.stop_workers()
        except Exception as e:
            logging.warning(f"Error deteniendo workers: {e}")
        
        try:
            if hasattr(self.visualizer, 'stop_visualization') and self.visualizer.stop_visualization:
                self.visualizer.stop_visualization()
        except Exception as e:
            logging.warning(f"Error deteniendo visualizador: {e}")
        
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logging.warning(f"Error cerrando socket: {e}")
        
        # Limpieza final de memoria
        self._memory_cleanup()
        
        if self.start_time:
            run_time = time.time() - self.start_time
            processing_rate = self.processing_count / run_time if run_time > 0 else 0
            
            logging.info(f"\n{'='*60}")
            logging.info(f"REPORTE FINAL - MONITOR CREIME_RT")
            logging.info(f"{'='*60}")
            logging.info(f"Monitor en Tiempo Real:")
            logging.info(f"  Inicio: {self.monitor_start_time}")
            logging.info(f"  Duración: {run_time:.1f} segundos")
            logging.info(f"")
            logging.info(f"Rendimiento del Sistema:")
            logging.info(f"  Paquetes procesados: {self.packet_count}")
            logging.info(f"  Ventanas CREIME_RT: {self.processing_count}")
            logging.info(f"  Tasa procesamiento: {processing_rate:.2f} ventanas/segundo")
            logging.info(f"")
            
            # Tiempos de detección
            if self.first_detection_time:
                logging.info(f"Primera Detección: {self.first_detection_time}")
            if self.first_confirmation_time:
                logging.info(f"Primera Confirmación: {self.first_confirmation_time}")
            
            logging.info(f"Anomalías Detectadas: {self.anomaly_count}")
            logging.info(f"Sismos Confirmados: {self.confirmed_earthquake_count}")
            logging.info(f"Total Eventos: {len(self.detected_events)}")
            
            # Estadísticas de valores CREIME_RT para diagnóstico
            if hasattr(self, 'creime_values') and self.creime_values:
                min_val = min(self.creime_values)
                max_val = max(self.creime_values)
                mean_val = sum(self.creime_values) / len(self.creime_values)
                above_threshold = sum(1 for v in self.creime_values if v > self.detection_threshold)
                logging.info(f"")
                logging.info(f"Estadísticas CREIME_RT:")
                logging.info(f"  Valores mínimo/máximo: {min_val:.2f} / {max_val:.2f}")
                logging.info(f"  Valor promedio: {mean_val:.2f}")
                logging.info(f"  Ventanas > umbral ({self.detection_threshold}): {above_threshold}/{len(self.creime_values)}")
                logging.info(f"  Porcentaje activación: {(above_threshold/len(self.creime_values)*100):.2f}%")
            
            if self.detected_events:
                # Calcular magnitud máxima de sismos confirmados
                max_magnitude = 0.0
                confirmed_events = [e for e in self.detected_events if e['type'] == 'confirmed_earthquake']
                
                for event in confirmed_events:
                    if 'magnitude' in event and event['magnitude']:
                        max_magnitude = max(max_magnitude, event['magnitude'])
                
                if max_magnitude > 0:
                    logging.info(f"Magnitud Final (Máxima): {max_magnitude:.2f}")
                
                logging.info(f"")
                logging.info(f"Detalle de Eventos:")
                
                # Agrupar por tipo
                anomalies = [e for e in self.detected_events if e['type'] == 'anomaly']
                earthquakes = [e for e in self.detected_events if e['type'] == 'confirmed_earthquake']
                
                if anomalies:
                    logging.info(f"  Anomalías Detectadas ({len(anomalies)}):")
                    for i, event in enumerate(anomalies, 1):
                        logging.info(f"    {i}. Tiempo: {event['event_time']}")
                        logging.info(f"       Confianza: {event['confidence']:.2f}")
                        logging.info(f"       Consecutivas: {event['consecutive_anomalies']}/3")
                
                if earthquakes:
                    logging.info(f"  Sismos Confirmados ({len(earthquakes)}):")
                    for i, event in enumerate(earthquakes, 1):
                        logging.info(f"    {i}. Tiempo: {event['event_time']}")
                        logging.info(f"       Confianza: {event['confidence']:.2f}")
                        logging.info(f"       Magnitud: {event['magnitude']:.2f}")
                        logging.info(f"       Ventana: {event['processing_id']}")
            else:
                logging.info(f"  No se detectaron eventos sísmicos")
            
            logging.info(f"{'='*60}")
            
            # Generar gráfico de raw output
            self.plot_creime_output_timeline()

def main():
    """Función principal del monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor CREIME_RT en tiempo real con AnyShake')
    parser.add_argument('--model_path', default='/home/aisaac/SAIPy/saipy/saved_models/', help='Directorio del modelo CREIME_RT')
    parser.add_argument('--host', default='localhost', help='Host de AnyShake Observer')
    parser.add_argument('--port', type=int, default=30000, help='Puerto de AnyShake Observer')
    parser.add_argument('--production', action='store_true', help='Modo producción con optimizaciones de memoria')
    
    args = parser.parse_args()
    
    # Buscar directorio con modelo CREIME_RT
    model_search_paths = [
        args.model_path,
        './saipy/saved_models/',
        '../saipy/saved_models/',
        '../../saipy/saved_models/',
        '/home/aisaac/SAIPy/saipy/saved_models/',
        './saved_models/',
        '../saved_models/',
        './models/',
        '../models/'
    ]
    
    model_dir = None
    for search_path in model_search_paths:
        if os.path.isdir(search_path):
            # Verificar que contenga CREIME_RT.h5 y CREIME_RT.json
            h5_file = os.path.join(search_path, 'CREIME_RT.h5')
            json_file = os.path.join(search_path, 'CREIME_RT.json')
            if os.path.isfile(h5_file) and os.path.isfile(json_file):
                model_dir = search_path
                break
    
    if not model_dir:
        logging.error(f"No se encontró directorio con modelo CREIME_RT completo")
        logging.error(f"Ubicaciones buscadas: {model_search_paths}")
        logging.error(f"Se requieren archivos: CREIME_RT.h5 y CREIME_RT.json")
        sys.exit(1)
    
    logging.info(f"Directorio del modelo encontrado: {model_dir}")
    
    # Crear directorios
    for directory in ["logs", "events_monitor"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Verificar SAIPy
    try:
        import saipy
    except ImportError:
        logging.error("SAIPy no disponible")
        sys.exit(1)
    
    # Configurar modo producción si está habilitado
    if args.production:
        logging.info("MODO PRODUCCIÓN ACTIVADO - Optimizaciones de memoria habilitadas")
        # Reducir buffers de visualización para producción
        global MAX_VISUALIZATION_SAMPLES, DISPLAY_SECONDS
        DISPLAY_SECONDS = 30  # Reducir a 30 segundos
        MAX_VISUALIZATION_SAMPLES = DISPLAY_SECONDS * SAMPLING_RATE
    
    # Crear monitor con directorio del modelo
    monitor = RealTimeMonitor(
        model_path=model_dir,
        host=args.host,
        port=args.port
    )
    
    # Configurar modo producción en el monitor
    if args.production:
        monitor.production_mode = True
    
    try:
        if monitor.start_monitor():
            if VISUALIZATION_ENABLED:
                plt.show(block=True)
            else:
                while monitor.running:
                    time.sleep(1)
        else:
            logging.error("Fallo en inicio del monitor")
            
    except KeyboardInterrupt:
        logging.info("Monitor detenido por usuario")
    except Exception as e:
        logging.error(f"Error crítico en monitor: {e}")
    finally:
        monitor.stop_monitor()

if __name__ == "__main__":
    main()
