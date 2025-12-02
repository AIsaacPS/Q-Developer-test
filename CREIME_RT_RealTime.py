#!/usr/bin/env python3
"""
CREIME_RT TIEMPO REAL - Basado en técnicas del simulador exitoso
Adaptación directa del simulador para TCP AnyShake con latencia <500ms
"""

import socket
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import logging
import os
from datetime import datetime
from scipy.signal import butter, filtfilt, lfilter
import sys
import gc
import psutil
import weakref
from threading import RLock

# Configuración GPU optimizada (copiada del simulador)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    from saipy.models.creime import CREIME_RT
    SAIPY_AVAILABLE = True
except ImportError:
    SAIPY_AVAILABLE = False
    logging.error("SAIPy no disponible")
    sys.exit(1)

class UltraFastBuffer:
    """Buffer ultra-rápido - COPIADO EXACTO del simulador exitoso"""
    
    def __init__(self, window_size=1500, sampling_rate=100, update_interval=0.1):
        self.window_size = int(window_size)  # 15 segundos (1500 muestras) - CREIME_RT hará padding
        self.sampling_rate = int(sampling_rate)
        self.update_interval = update_interval  # 0.1 segundos como simulador
        
        total_size = self.window_size + 500  # Buffer para 20 segundos
        self.buffers = {
            'ENZ': deque(maxlen=total_size),
            'ENE': deque(maxlen=total_size),
            'ENN': deque(maxlen=total_size)
        }
        
        self.lock = threading.Lock()
        self.window_count = 0
        self.last_window_time = 0
        self.ready = False
        self.min_ready_samples = int(15 * sampling_rate)  # 15 segundos mínimos para primera ventana
        self.new_data_event = threading.Event()
        
        # Buffer para serie temporal de CREIME_RT (ultra-optimizado para memoria)
        self.creime_results = deque(maxlen=300)  # 5 minutos a 1Hz - reducido para estabilidad
        self.creime_timestamps = deque(maxlen=300)
        
        # Contadores para limpieza periódica agresiva
        self.cleanup_counter = 0
        self.cleanup_interval = 3600  # Cada hora para estabilidad 24/7
        self.last_cleanup_time = time.time()
        
        # Monitoreo de memoria
        self.memory_threshold = 1.5 * 1024 * 1024 * 1024  # 1.5GB límite
        self.last_memory_check = time.time()
        
        # Pre-calcular filtro (TÉCNICA CLAVE DEL SIMULADOR) - 1-30 Hz
        self.filter_b, self.filter_a = butter(4, [1.0/50, 30.0/50], btype='band')
        self.filter_zi = {
            'ENZ': np.zeros(max(len(self.filter_a), len(self.filter_b)) - 1),
            'ENE': np.zeros(max(len(self.filter_a), len(self.filter_b)) - 1),
            'ENN': np.zeros(max(len(self.filter_a), len(self.filter_b)) - 1)
        }
        
    def add_data(self, component, data, timestamp):
        """Añade datos con filtrado en tiempo real - TÉCNICA DEL SIMULADOR"""
        with self.lock:
            if component in self.buffers:
                # Aplicar filtro en tiempo real usando lfilter con zi
                try:
                    filtered_data, self.filter_zi[component] = lfilter(
                        self.filter_b, self.filter_a, data, zi=self.filter_zi[component]
                    )
                    self.buffers[component].extend(filtered_data.tolist())
                except:
                    # Fallback sin filtro
                    self.buffers[component].extend(data)
                
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
        """Genera ventana deslizante optimizada para estabilidad 24/7"""
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_window_time < self.update_interval:
                return None
                
            # Verificar que tenemos suficientes datos para ventana de 15 segundos
            min_samples = min(len(buf) for buf in self.buffers.values())
            if min_samples < self.window_size:
                return None
            
            try:
                # Usar numpy arrays directamente para eficiencia
                window_arrays = []
                for component in ['ENZ', 'ENE', 'ENN']:
                    buf = self.buffers[component]
                    buf_len = len(buf)
                    
                    # Extraer ventana de 15 segundos (1500 muestras)
                    if buf_len < self.window_size:
                        # Padding si no hay suficientes datos
                        component_data_15s = np.pad(np.array(list(buf)), 
                                                   (self.window_size - buf_len, 0), 
                                                   mode='constant', constant_values=0.0)
                    else:
                        # Slice de los últimos 15 segundos
                        start_idx = buf_len - self.window_size
                        component_data_15s = np.array([buf[i] for i in range(start_idx, buf_len)])
                    
                    # Normalización z-score en ventana de 15s
                    if len(component_data_15s) > 1:
                        mean_val = np.mean(component_data_15s)
                        std_val = np.std(component_data_15s)
                        if std_val > 1e-8:
                            component_data_15s = (component_data_15s - mean_val) / std_val
                        else:
                            component_data_15s.fill(0.0)
                    
                    # CREIME_RT hará el padding automáticamente - enviamos solo 15s
                    window_arrays.append(component_data_15s)
                
                # Stack directo - formato [1, 1500, 3] - CREIME_RT hará padding a [1, 3000, 3]
                window_3d = np.stack(window_arrays, axis=1)
                window_3d = np.expand_dims(window_3d, axis=0).astype(np.float32)
                
                self.window_count += 1
                self.last_window_time = current_time
                
                # Log cada 100 ventanas para mostrar que CREIME_RT hará el padding
                if self.window_count % 100 == 0:
                    logging.info(
                        f"Ventana #{self.window_count}: Enviando 15s a CREIME_RT - El modelo hará padding automático a 30s | "
                        f"Shape enviada: {window_3d.shape} | "
                        f"Datos disponibles: {min_samples} muestras"
                    )
                
                return window_3d
                
            except Exception as e:
                logging.error(f"Error creando ventana 3D: {e}")
                return None
    
    def add_creime_result(self, raw_output):
        """Añade resultado de CREIME_RT con monitoreo de memoria continuo"""
        with self.lock:
            self.creime_results.append(raw_output)
            self.creime_timestamps.append(time.time())
            
            # Limpieza periódica más frecuente
            self.cleanup_counter += 1
            current_time = time.time()
            
            # Limpieza cada hora O si detectamos uso alto de memoria
            if (self.cleanup_counter >= self.cleanup_interval or 
                current_time - self.last_cleanup_time > 3600 or
                current_time - self.last_memory_check > 300):  # Check cada 5 min
                
                self._cleanup_memory()
                self.cleanup_counter = 0
                self.last_memory_check = current_time
    
    def _cleanup_memory(self):
        """Limpieza agresiva de memoria para operación 24/7"""
        try:
            # Limpieza completa de garbage collection
            for generation in range(3):
                collected = gc.collect(generation)
            
            # Verificar uso de memoria
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Limpieza adicional si excede umbral
            if memory_mb > 1500:  # 1.5GB
                # Reducir buffers temporalmente
                if len(self.creime_results) > 150:
                    for _ in range(50):
                        if self.creime_results:
                            self.creime_results.popleft()
                        if self.creime_timestamps:
                            self.creime_timestamps.popleft()
                
                # Forzar limpieza adicional
                gc.collect()
                
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                logging.warning(f"Limpieza memoria crítica: {memory_mb:.1f}MB → {memory_mb_after:.1f}MB")
            else:
                logging.info(f"Limpieza memoria: {memory_mb:.1f}MB - OK")
            
            self.last_cleanup_time = time.time()
            
        except Exception as e:
            logging.error(f"Error en limpieza de memoria: {e}")
    
    def get_visualization_data(self):
        """Datos para visualización incluyendo serie temporal de CREIME_RT"""
        with self.lock:
            # Obtener longitud mínima disponible
            min_len = min(len(self.buffers[comp]) for comp in ['ENZ', 'ENE', 'ENN'])
            
            if min_len == 0:
                return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
            
            # Tomar hasta 1500 muestras (15 segundos) o las disponibles
            samples_to_take = min(min_len, 1500)
            
            # Serie temporal de CREIME_RT con timestamps
            creime_data = np.array(list(self.creime_results)) if self.creime_results else np.array([])
            creime_timestamps = np.array(list(self.creime_timestamps)) if self.creime_timestamps else np.array([])
            
            return (np.array(list(self.buffers['ENZ'])[-samples_to_take:]), 
                   np.array(list(self.buffers['ENE'])[-samples_to_take:]), 
                   np.array(list(self.buffers['ENN'])[-samples_to_take:]),
                   creime_data,
                   creime_timestamps)

class UltraFastProcessingPipeline:
    """Pipeline de procesamiento optimizado para estabilidad 24/7"""
    
    def __init__(self, model_path, num_workers=1):
        self.model_path = model_path
        self.num_workers = num_workers
        self.processing_queue = queue.Queue(maxsize=3)  # Ligeramente mayor para estabilidad
        self.result_queue = queue.Queue(maxsize=5)  # Limitar acumulación
        self.workers = []
        self.running = False
        
        # Monitoreo de workers para recuperación automática
        self.worker_health = {}
        self.last_health_check = time.time()
        self.worker_restart_count = 0
        
        self.start_workers()
    
    def start_workers(self):
        """Inicia workers de procesamiento"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self.processing_worker,
                name=f"CREIME_RealTime_Worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"Iniciados {self.num_workers} workers tiempo real")
    
    def processing_worker(self):
        """Worker robusto para CREIME_RT con recuperación automática"""
        worker_id = threading.current_thread().name
        self.worker_health[worker_id] = {'last_activity': time.time(), 'error_count': 0}
        
        try:
            model = CREIME_RT(self.model_path)
            logging.info(f"Worker {worker_id} CREIME_RT inicializado")
        except Exception as e:
            logging.error(f"Worker {worker_id} no pudo cargar modelo: {e}")
            return
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                window_data, processing_id = self.processing_queue.get(timeout=2.0)
                
                if window_data is None:
                    break
                
                # Actualizar salud del worker
                self.worker_health[worker_id]['last_activity'] = time.time()
                
                start_time = time.time()
                
                try:
                    # PREDICCIÓN con manejo robusto de errores
                    y_pred, predictions = model.predict(window_data)
                    
                    # Usar el valor medio del vector completo
                    if y_pred is not None and len(y_pred.shape) > 1:
                        raw_output = round(float(np.mean(y_pred[0])), 1)
                    else:
                        raw_output = -4.0
                    
                    # Nueva clasificación de eventos con umbral de sismo en 2.0
                    if raw_output > -0.5:  # Anomalía detectada
                        detection = 1
                        if raw_output > 2.0:  # Sismo confirmado (umbral subido a 2.0)
                            magnitude = round(4.0 + raw_output, 1)  # Magnitud local redondeada
                            event_type = "SISMO_CONFIRMADO"
                        else:
                            magnitude = None
                            event_type = "ANOMALIA_DETECTADA"
                    else:
                        detection = 0
                        magnitude = None
                        event_type = "RUIDO"
                    
                    result = (detection, magnitude, raw_output, event_type)
                    consecutive_errors = 0  # Reset en éxito
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.worker_health[worker_id]['error_count'] += 1
                    logging.warning(f"Error en predicción worker {worker_id}: {e} (consecutivos: {consecutive_errors})")
                    result = (0, None, -4.0, "ERROR")
                    raw_output = -4.0
                    
                    # Si hay muchos errores consecutivos, reiniciar worker
                    if consecutive_errors >= max_consecutive_errors:
                        logging.error(f"Worker {worker_id} con {consecutive_errors} errores consecutivos, reiniciando")
                        break
                
                processing_time = time.time() - start_time
                
                # Evitar acumulación en result_queue
                try:
                    self.result_queue.put({
                        'result': result,
                        'processing_id': processing_id,
                        'processing_time': processing_time,
                        'raw_output': raw_output,
                        'event_type': result[3] if len(result) > 3 else "UNKNOWN"
                    }, timeout=0.1)
                except queue.Full:
                    logging.warning(f"Result queue llena, descartando resultado {processing_id}")
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                # Verificar salud periódicamente durante timeout
                self._check_worker_health()
                continue
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"Error crítico en worker {worker_id}: {e}")
                try:
                    self.processing_queue.task_done()
                except:
                    pass
                
                if consecutive_errors >= max_consecutive_errors:
                    break
                    
                time.sleep(0.1)
        
        logging.warning(f"Worker {worker_id} terminando")
        
    def _check_worker_health(self):
        """Verifica salud de workers y reinicia si es necesario"""
        current_time = time.time()
        
        if current_time - self.last_health_check < 60:  # Check cada minuto
            return
            
        dead_workers = []
        for worker_id, health in self.worker_health.items():
            if current_time - health['last_activity'] > 120:  # 2 minutos sin actividad
                dead_workers.append(worker_id)
        
        if dead_workers:
            logging.warning(f"Workers inactivos detectados: {dead_workers}")
            self._restart_dead_workers(dead_workers)
        
        self.last_health_check = current_time
    
    def _restart_dead_workers(self, dead_worker_ids):
        """Reinicia workers muertos"""
        for worker_id in dead_worker_ids:
            if worker_id in self.worker_health:
                del self.worker_health[worker_id]
        
        # Contar workers vivos
        alive_workers = len([w for w in self.workers if w.is_alive()])
        
        if alive_workers < self.num_workers:
            workers_to_start = self.num_workers - alive_workers
            logging.info(f"Reiniciando {workers_to_start} workers")
            
            for i in range(workers_to_start):
                self.worker_restart_count += 1
                worker = threading.Thread(
                    target=self.processing_worker,
                    name=f"CREIME_RealTime_Worker_Restart_{self.worker_restart_count}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
    
    def submit_window(self, window_data, processing_id):
        """Envía ventana para procesamiento"""
        try:
            self.processing_queue.put((window_data, processing_id), timeout=0.1)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout=0.3):
        """Obtiene resultado con limpieza agresiva de queue para estabilidad 24/7"""
        try:
            result = self.result_queue.get(timeout=timeout)
            
            # Limpieza más agresiva para prevenir acumulación
            queue_size = self.result_queue.qsize()
            if queue_size > 3:  # Umbral más bajo
                logging.warning(f"Queue de resultados acumulada ({queue_size}), limpiando")
                cleaned = 0
                while not self.result_queue.empty() and cleaned < queue_size - 1:
                    try:
                        self.result_queue.get_nowait()
                        cleaned += 1
                    except queue.Empty:
                        break
                logging.info(f"Limpiados {cleaned} resultados de queue")
            
            return result
        except queue.Empty:
            return None
    
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

class AnyShakeConnector:
    """Conector AnyShake ultra-robusto para operación 24/7"""
    
    def __init__(self, host='localhost', port=30000):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.packet_count = 0
        self.last_data_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10  # Más intentos para estabilidad
        
        # Estadísticas para monitoreo 24/7
        self.total_reconnections = 0
        self.connection_start_time = None
        self.bytes_received = 0
        self.last_stats_log = time.time()
        
    def connect(self):
        try:
            if self.sock:
                self.sock.close()
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # Timeout más generoso
            
            # Configuraciones de socket para estabilidad
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            
            self.sock.connect((self.host, self.port))
            self.sock.sendall(b"AT+REALTIME=1\r\n")
            time.sleep(2)  # Más tiempo para estabilizar
            self.sock.setblocking(False)
            
            self.connected = True
            self.connection_start_time = time.time()
            self.last_data_time = time.time()
            
            logging.info("✅ Conectado a AnyShake Observer con configuración robusta")
            return True
        except Exception as e:
            logging.error(f"Error de conexión: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    def parse_packet(self, data_line):
        """Parser ultra-minimalista"""
        try:
            if not data_line.startswith('$') or '*' not in data_line:
                return None
                
            parts = data_line.split(',')
            if len(parts) < 8:
                return None
            
            component = parts[4]
            data_values = []
            
            for i in range(7, len(parts)):
                part = parts[i].split('*')[0]  # Remover checksum
                if part.lstrip('-').isdigit():
                    data_values.append(int(part))
            
            return {'component': component, 'data': data_values} if data_values else None
        except:
            return None
    
    def read_data(self):
        if not self.connected:
            return []
            
        try:
            raw_data = self.sock.recv(8192)  # Buffer más grande para eficiencia
            if raw_data:
                self.last_data_time = time.time()
                self.reconnect_attempts = 0
                self.bytes_received += len(raw_data)
                
                # Log estadísticas periódicamente
                self._log_connection_stats()
                
                lines = raw_data.decode('utf-8', errors='ignore').split('\r')
                packets = []
                for line in lines:
                    if line.strip():
                        parsed = self.parse_packet(line.strip())
                        if parsed:
                            packets.append(parsed)
                            self.packet_count += 1
                return packets
            else:
                # Socket cerrado por el servidor
                logging.warning("Socket cerrado por servidor")
                self.connected = False
                self._attempt_reconnect()
                
        except BlockingIOError:
            # Verificar timeout de datos más agresivo
            if time.time() - self.last_data_time > 15:  # 15s sin datos
                logging.warning("Timeout de datos (15s), intentando reconexión")
                self._attempt_reconnect()
        except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
            logging.error(f"Error de conexión de red: {e}")
            self.connected = False
            self._attempt_reconnect()
        except Exception as e:
            logging.error(f"Error inesperado en lectura: {e}")
            self.connected = False
            self._attempt_reconnect()
            
        return []
    
    def _log_connection_stats(self):
        """Log estadísticas de conexión periódicamente"""
        current_time = time.time()
        if current_time - self.last_stats_log > 3600:  # Cada hora
            if self.connection_start_time:
                uptime = current_time - self.connection_start_time
                mb_received = self.bytes_received / (1024 * 1024)
                logging.info(
                    f"Estadísticas AnyShake - Uptime: {uptime/3600:.1f}h | "
                    f"Paquetes: {self.packet_count} | "
                    f"Datos: {mb_received:.1f}MB | "
                    f"Reconexiones: {self.total_reconnections}"
                )
            self.last_stats_log = current_time
    
    def _attempt_reconnect(self):
        """Intenta reconexión automática con backoff inteligente"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self.total_reconnections += 1
            
            # Backoff exponencial limitado
            delay = min(2 ** self.reconnect_attempts, 30)  # Máximo 30 segundos
            
            logging.info(
                f"Intento de reconexión {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                f"(total: {self.total_reconnections}) - Esperando {delay}s"
            )
            
            self.close()
            time.sleep(delay)
            
            if self.connect():
                logging.info("Reconexión exitosa")
            else:
                logging.warning(f"Reconexión {self.reconnect_attempts} falló")
        else:
            logging.error(f"Máximo de intentos de reconexión alcanzado ({self.max_reconnect_attempts})")
            # Reset después de un tiempo para permitir recuperación
            time.sleep(60)
            self.reconnect_attempts = 0
    
    def close(self):
        if self.sock:
            self.sock.close()
            self.connected = False

class RealTimeVisualizer:
    """Visualizador optimizado con subplot CREIME_RT"""
    
    def __init__(self, buffer_manager, processing_pipeline):
        self.buffer_manager = buffer_manager
        self.processing_pipeline = processing_pipeline
        self.latest_result = None
        self.event_count = 0
        
        # Buffer para raw output CREIME_RT
        self.creime_values = deque(maxlen=300)  # 30 segundos a 10 Hz
        self.creime_times = deque(maxlen=300)
        self.start_time = time.time()
        
        self.setup_plot()
        
    def setup_plot(self):
        # Configurar tema oscuro
        plt.style.use('dark_background')
        
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(12, 10))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.fig.suptitle('SKYALERT AI SEISMIC STATION 001', fontsize=14, fontweight='bold', color='white')
        
        # Señales sísmicas con colores claros uniformes
        seismic_color = '#00ff88'  # Verde claro brillante para todas las trazas
        self.line_enz, = self.ax1.plot([], [], color=seismic_color, linewidth=1.2)
        self.line_ene, = self.ax2.plot([], [], color=seismic_color, linewidth=1.2)
        self.line_enn, = self.ax3.plot([], [], color=seismic_color, linewidth=1.2)
        
        # CREIME_RT Raw Output con color distintivo
        self.line_creime, = self.ax4.plot([], [], '#ff6b35', linewidth=2.5, label='CREIME_RT Raw Output')
        
        # Configurar ejes sísmicos con tema oscuro
        seismic_axes = [self.ax1, self.ax2, self.ax3]
        labels = ['ENZ (Vertical)\nAcceleration', 'ENE (East)\nAcceleration', 'ENN (North)\nAcceleration']
        
        for ax, label in zip(seismic_axes, labels):
            ax.set_facecolor('#2d2d2d')
            ax.set_ylabel(label, color='white', fontsize=10)
            ax.set_xlim(-15, 0)  # Ventana de 15 segundos
            ax.grid(True, alpha=0.2, color='#555555')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('#666666')
            ax.spines['top'].set_color('#666666')
            ax.spines['right'].set_color('#666666')
            ax.spines['left'].set_color('#666666')
            
        # Configurar eje CREIME_RT con tema oscuro
        self.ax4.set_facecolor('#2d2d2d')
        self.ax4.set_xlim(-15, 0)  # Ventana de 15 segundos
        self.ax4.set_ylim(-5, 6)
        self.ax4.grid(True, alpha=0.2, color='#555555')
        self.ax4.set_ylabel('CREIME_RT\nRaw Output', color='white', fontsize=10)
        self.ax4.set_xlabel('Time (seconds ago) - 15s data (CREIME_RT auto-pads)', color='white', fontsize=9)
        self.ax4.tick_params(colors='white', labelsize=8)
        self.ax4.spines['bottom'].set_color('#666666')
        self.ax4.spines['top'].set_color('#666666')
        self.ax4.spines['right'].set_color('#666666')
        self.ax4.spines['left'].set_color('#666666')
        
        # Líneas de referencia CREIME_RT con nueva clasificación
        self.ax4.axhline(y=-0.5, color='#ffaa00', linestyle='--', alpha=0.8, linewidth=1.5, label='Umbral Anomalía (-0.5)')
        self.ax4.axhline(y=2.0, color='#ff4444', linestyle='--', alpha=0.8, linewidth=2.0, label='Umbral Sismo (2.0)')
        self.ax4.axhline(y=-4.0, color='#888888', linestyle='--', alpha=0.6, linewidth=1.0, label='Ruido Base')
        
        legend = self.ax4.legend(loc='upper right', fontsize=8)
        legend.get_frame().set_facecolor('#3d3d3d')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
    def update_plot(self, frame):
        try:
            # Obtener datos incluyendo serie temporal CREIME_RT
            enz_data, ene_data, enn_data, creime_data, creime_timestamps = self.buffer_manager.get_visualization_data()
            
            # Obtener último resultado
            result = self.processing_pipeline.get_result(timeout=0.01)
            if result:
                self.latest_result = result
                
                # Log de salida cruda CREIME_RT SIEMPRE
                raw_output = result['result'][2]
                processing_time = result['processing_time']
                logging.info(f"CREIME_RT Raw Output: {raw_output:.1f} | Latencia: {processing_time:.3f}s")
                
                # Los datos ya se almacenan en el buffer principal
                
                # Verificar detección con nueva clasificación
                if result['result'][0] == 1:  # Detección
                    event_type = result.get('event_type', 'UNKNOWN')
                    magnitude = result['result'][1]
                    
                    if event_type == "SISMO_CONFIRMADO":
                        self.event_count += 1
                        logging.critical(
                            f"🚨 SISMO CONFIRMADO 🚨\n"
                            f"Raw Output: {raw_output:.1f}\n"
                            f"Magnitud Local: {magnitude:.1f}\n"
                            f"Latencia: {processing_time:.3f}s"
                        )
                    elif event_type == "ANOMALIA_DETECTADA":
                        logging.warning(
                            f"⚠️ ANOMALÍA DETECTADA ⚠️\n"
                            f"Raw Output: {raw_output:.1f}\n"
                            f"Umbral superado pero sin confirmación de sismo\n"
                            f"Latencia: {processing_time:.3f}s"
                        )
            
            # Actualizar gráficos sísmicos - siempre mostrar últimos 15 segundos
            if len(enz_data) > 0:
                # Siempre mostrar exactamente los últimos 15 segundos (1500 muestras)
                samples_to_show = 1500
                time_axis = np.linspace(-15, 0, samples_to_show)
                
                # Tomar exactamente los últimos 15 segundos, rellenar con ceros si es necesario
                if len(enz_data) >= samples_to_show:
                    enz_display = enz_data[-samples_to_show:]
                    ene_display = ene_data[-samples_to_show:]
                    enn_display = enn_data[-samples_to_show:]
                else:
                    # Rellenar con ceros al inicio si no hay suficientes datos
                    padding = samples_to_show - len(enz_data)
                    enz_display = np.pad(enz_data, (padding, 0), mode='constant', constant_values=0)
                    ene_display = np.pad(ene_data, (padding, 0), mode='constant', constant_values=0)
                    enn_display = np.pad(enn_data, (padding, 0), mode='constant', constant_values=0)
                
                self.line_enz.set_data(time_axis, enz_display)
                self.line_ene.set_data(time_axis, ene_display)
                self.line_enn.set_data(time_axis, enn_display)
                
                # Auto-escalar ejes Y para ventana de 15 segundos
                for ax, data in zip([self.ax1, self.ax2, self.ax3], [enz_display, ene_display, enn_display]):
                    if len(data) > 0:
                        y_range = np.max(data) - np.min(data)
                        if y_range > 0:
                            margin = y_range * 0.1
                            ax.set_ylim(np.min(data) - margin, np.max(data) + margin)
                        else:
                            ax.set_ylim(-10, 10)
            
            # Actualizar gráfico CREIME_RT - siempre mostrar últimos 15 segundos
            if len(creime_data) > 0 and len(creime_timestamps) > 0:
                current_time = time.time()
                # Tomar exactamente los últimos 15 segundos de datos CREIME_RT
                samples_creime = min(len(creime_data), 150)  # 15 segundos a ~10Hz
                
                if samples_creime > 0:
                    recent_creime_data = creime_data[-samples_creime:]
                    recent_timestamps = creime_timestamps[-samples_creime:]
                    
                    # Crear eje de tiempo uniforme para 15 segundos
                    if len(recent_timestamps) > 1:
                        creime_time_axis = [-(current_time - t) for t in recent_timestamps]
                        # Limitar a ventana de 15 segundos
                        valid_mask = [t >= -15 for t in creime_time_axis]
                        if any(valid_mask):
                            creime_time_axis = np.array(creime_time_axis)[valid_mask]
                            recent_creime_data = np.array(recent_creime_data)[valid_mask]
                            self.line_creime.set_data(creime_time_axis, recent_creime_data)
            
            # Actualizar título con tema oscuro - ventana de 15s
            title = f'SKYALERT AI SEISMIC STATION 001 - Eventos: {self.event_count}'
            
            if self.latest_result and 'result' in self.latest_result and 'processing_time' in self.latest_result:
                try:
                    raw_val = self.latest_result["result"][2]
                    proc_time = self.latest_result["processing_time"]
                    title += f' | Raw: {raw_val:.2f} | {proc_time:.3f}s'
                except (IndexError, TypeError, KeyError):
                    pass
                
            self.fig.suptitle(title, fontsize=12, color='white')
            
        except Exception as e:
            logging.error(f"Error en visualización: {e}")
            
        return [self.line_enz, self.line_ene, self.line_enn, self.line_creime]

class CreimeRTRealTimeMonitor:
    """Sistema principal - ARQUITECTURA DEL SIMULADOR ADAPTADA"""
    
    def __init__(self, host='localhost', port=30000, model_path=None, no_gui=False):
        # Configurar path del modelo como en simulador
        if model_path is None:
            import saipy
            saipy_path = os.path.dirname(saipy.__file__)
            model_path = os.path.join(saipy_path, 'saved_models', '')
            if not model_path.endswith('/'):
                model_path += '/'
        
        # Componentes principales optimizados para 24/7 - PRUEBA 15 SEGUNDOS
        self.buffer = UltraFastBuffer(
            window_size=1500,  # 15 segundos - PRUEBA REDUCIDA
            sampling_rate=100,
            update_interval=0.1  # 0.1 segundos como simulador
        )
        
        self.processing_pipeline = UltraFastProcessingPipeline(model_path, num_workers=1)
        self.connector = AnyShakeConnector(host, port)
        
        # Visualizador opcional para modo servidor
        self.no_gui = no_gui
        self.visualizer = None if no_gui else RealTimeVisualizer(self.buffer, self.processing_pipeline)
        
        # Estado del sistema
        self.running = False
        self.detection_count = 0
        self.processing_count = 0
        self.last_processing_time = 0
        self.latency_target = 0.1
        
        # Threads del sistema
        self.data_thread = None
        self.processing_thread = None
        self.health_thread = None
        
        # Monitoreo avanzado de salud del sistema para 24/7
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutos
        self.error_count = 0
        self.max_errors_per_hour = 50
        
        # Métricas adicionales para estabilidad
        self.memory_usage_history = deque(maxlen=288)  # 24 horas a 5 min
        self.cpu_usage_history = deque(maxlen=288)
        self.last_performance_log = time.time()
        
        # Control de degradación
        self.degraded_mode = False
        self.degraded_mode_start = None
        
        # Crear directorios necesarios
        os.makedirs('logs', exist_ok=True)
        os.makedirs('events_monitor', exist_ok=True)
        
    def start_monitoring(self):
        """Inicia monitoreo robusto para operación 24/7"""
        logging.info("🚀 Iniciando CREIME_RT MONITOR 24/7...")
        
        # Intentar conexión con reintentos
        max_connection_attempts = 5
        for attempt in range(max_connection_attempts):
            if self.connector.connect():
                break
            logging.warning(f"Intento de conexión {attempt + 1}/{max_connection_attempts} falló")
            if attempt < max_connection_attempts - 1:
                time.sleep(5)
        else:
            logging.error("❌ No se pudo conectar a AnyShake Observer después de múltiples intentos")
            return False
        
        self.running = True
        
        # Iniciar threads del sistema
        self.data_thread = threading.Thread(target=self._data_acquisition_loop, daemon=True, name="DataAcquisition")
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True, name="Processing")
        self.health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True, name="HealthMonitor")
        
        self.data_thread.start()
        self.processing_thread.start()
        self.health_thread.start()
        
        logging.info("✅ Sistema iniciado - Todos los threads activos")
        
        if self.no_gui:
            # Modo servidor sin GUI
            logging.info("Ejecutando en modo servidor (sin GUI)")
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("🛑 Señal de interrupción recibida")
        else:
            # Modo con visualización
            try:
                ani = animation.FuncAnimation(
                    self.visualizer.fig, 
                    self.visualizer.update_plot, 
                    interval=200,  # Ligeramente más lento para estabilidad
                    blit=True, 
                    cache_frame_data=False
                )
                plt.show()
            except KeyboardInterrupt:
                logging.info("🛑 Deteniendo sistema...")
            except Exception as e:
                logging.error(f"Error en visualización: {e}")
                logging.info("Continuando en modo servidor...")
                # Continuar sin GUI si falla la visualización
                try:
                    while self.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        
        self.stop_monitoring()
        return True
    
    def _health_monitoring_loop(self):
        """Loop dedicado para monitoreo de salud del sistema"""
        while self.running:
            try:
                self._check_system_health()
                time.sleep(60)  # Check cada minuto
            except Exception as e:
                logging.error(f"Error en monitoreo de salud: {e}")
                time.sleep(60)
    
    def _data_acquisition_loop(self):
        """Loop de adquisición robusto para operación 24/7"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                packets = self.connector.read_data()
                
                if packets:
                    consecutive_errors = 0  # Reset en datos exitosos
                    
                    for packet in packets:
                        # Conversión mg → Gals (como simulador)
                        converted_data = [x * 0.119 for x in packet['data']]
                        
                        # Usar mismo timestamp para buffer
                        current_time = time.time()
                        self.buffer.add_data(packet['component'], converted_data, current_time)
                
                time.sleep(0.01)  # 10ms como simulador
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                
                if consecutive_errors <= 3:
                    logging.warning(f"Error en adquisición ({consecutive_errors}): {e}")
                elif consecutive_errors == max_consecutive_errors:
                    logging.error(f"Demasiados errores consecutivos en adquisición ({consecutive_errors}), pausando")
                    time.sleep(5)  # Pausa más larga
                    consecutive_errors = 0  # Reset para intentar de nuevo
                
                time.sleep(0.1 if consecutive_errors < 5 else 1.0)
    
    def _check_system_health(self):
        """Monitoreo avanzado de salud del sistema para operación 24/7"""
        current_time = time.time()
        
        if current_time - self.last_health_check >= self.health_check_interval:
            try:
                # Métricas básicas
                uptime = current_time - self.start_time
                error_rate = self.error_count / max(uptime / 3600, 0.1)  # Evitar división por cero
                
                # Métricas de sistema
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                # Almacenar historial
                self.memory_usage_history.append(memory_mb)
                self.cpu_usage_history.append(cpu_percent)
                
                # Verificar condiciones de degradación
                self._check_degradation_conditions(memory_mb, cpu_percent, error_rate)
                
                # Log completo cada hora
                if current_time - self.last_performance_log > 3600:
                    self._log_detailed_performance(uptime, memory_mb, cpu_percent, error_rate)
                    self.last_performance_log = current_time
                else:
                    # Log básico cada 5 minutos
                    logging.info(
                        f"Sistema OK - Uptime: {uptime/3600:.1f}h | "
                        f"Memoria: {memory_mb:.1f}MB | "
                        f"CPU: {cpu_percent:.1f}% | "
                        f"Ventanas: {self.processing_count} | "
                        f"Detecciones: {self.detection_count}"
                    )
                
                # Gestión de errores
                if error_rate > self.max_errors_per_hour:
                    logging.warning(f"Tasa de errores alta: {error_rate:.1f}/h, activando modo degradado")
                    self._enter_degraded_mode()
                    self.error_count = 0
                
                self.last_health_check = current_time
                
            except Exception as e:
                logging.error(f"Error en monitoreo de salud: {e}")
    
    def _check_degradation_conditions(self, memory_mb, cpu_percent, error_rate):
        """Verifica condiciones que requieren modo degradado"""
        degradation_reasons = []
        
        # Memoria alta sostenida
        if len(self.memory_usage_history) >= 6:  # 30 minutos de datos
            avg_memory = sum(list(self.memory_usage_history)[-6:]) / 6
            if avg_memory > 1800:  # 1.8GB promedio
                degradation_reasons.append(f"Memoria alta sostenida: {avg_memory:.1f}MB")
        
        # CPU alta sostenida
        if len(self.cpu_usage_history) >= 6:
            avg_cpu = sum(list(self.cpu_usage_history)[-6:]) / 6
            if avg_cpu > 80:
                degradation_reasons.append(f"CPU alta sostenida: {avg_cpu:.1f}%")
        
        # Tasa de errores
        if error_rate > self.max_errors_per_hour * 0.8:
            degradation_reasons.append(f"Tasa de errores elevada: {error_rate:.1f}/h")
        
        if degradation_reasons and not self.degraded_mode:
            logging.warning(f"Activando modo degradado: {'; '.join(degradation_reasons)}")
            self._enter_degraded_mode()
        elif not degradation_reasons and self.degraded_mode:
            self._exit_degraded_mode()
    
    def _enter_degraded_mode(self):
        """Activa modo degradado para preservar estabilidad"""
        self.degraded_mode = True
        self.degraded_mode_start = time.time()
        
        # Reducir frecuencia de procesamiento
        self.buffer.update_interval = 0.2  # De 0.1s a 0.2s
        
        # Forzar limpieza de memoria
        self.buffer._cleanup_memory()
        
        logging.warning("MODO DEGRADADO ACTIVADO - Reduciendo carga del sistema")
    
    def _exit_degraded_mode(self):
        """Sale del modo degradado"""
        if self.degraded_mode:
            degraded_duration = time.time() - self.degraded_mode_start
            self.degraded_mode = False
            self.degraded_mode_start = None
            
            # Restaurar frecuencia normal
            self.buffer.update_interval = 0.1
            
            logging.info(f"MODO DEGRADADO DESACTIVADO - Duración: {degraded_duration/60:.1f} minutos")
    
    def _log_detailed_performance(self, uptime, memory_mb, cpu_percent, error_rate):
        """Log detallado de rendimiento cada hora"""
        # Estadísticas de memoria
        if self.memory_usage_history:
            mem_avg = sum(self.memory_usage_history) / len(self.memory_usage_history)
            mem_max = max(self.memory_usage_history)
        else:
            mem_avg = mem_max = memory_mb
        
        # Estadísticas de CPU
        if self.cpu_usage_history:
            cpu_avg = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            cpu_max = max(self.cpu_usage_history)
        else:
            cpu_avg = cpu_max = cpu_percent
        
        logging.info(
            f"\n=== REPORTE HORARIO DE RENDIMIENTO ===\n"
            f"Uptime: {uptime/3600:.1f} horas\n"
            f"Memoria - Actual: {memory_mb:.1f}MB | Promedio: {mem_avg:.1f}MB | Máximo: {mem_max:.1f}MB\n"
            f"CPU - Actual: {cpu_percent:.1f}% | Promedio: {cpu_avg:.1f}% | Máximo: {cpu_max:.1f}%\n"
            f"Procesamiento - Ventanas: {self.processing_count} | Detecciones: {self.detection_count}\n"
            f"Red - Paquetes: {self.connector.packet_count} | Reconexiones: {self.connector.total_reconnections}\n"
            f"Errores - Total: {self.error_count} | Tasa: {error_rate:.1f}/h\n"
            f"Modo degradado: {'SÍ' if self.degraded_mode else 'NO'}\n"
            f"======================================="
        )
    
    def _processing_loop(self):
        """Loop de procesamiento robusto con exportación de eventos"""
        self.last_processing_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                if self.buffer.wait_for_new_data(timeout=0.5):
                    self.buffer.reset_data_event()
                    
                    # Generar ventana deslizante de 30 segundos
                    window_data = self.buffer.get_latest_window()
                    
                    if window_data is not None:
                        processing_id = self.processing_count
                        
                        if self.processing_pipeline.submit_window(window_data, processing_id):
                            self.processing_count += 1
                            self.last_processing_time = time.time()
                            
                            # Obtener resultado
                            result_data = self.processing_pipeline.get_result(timeout=0.4)
                            
                            if result_data:
                                raw_output = result_data['raw_output']
                                
                                # Almacenar resultado para serie temporal
                                self.buffer.add_creime_result(raw_output)
                                
                                # Evaluar detección y exportar solo sismos confirmados
                                if result_data['result'][0] == 1:
                                    event_type = result_data.get('event_type', 'UNKNOWN')
                                    if event_type == "SISMO_CONFIRMADO":
                                        self.detection_count += 1
                                        self._export_seismic_event(result_data, window_data)
                                    elif event_type == "ANOMALIA_DETECTADA":
                                        logging.info(f"Anomalía detectada pero no confirmada como sismo")
                                
                                consecutive_errors = 0  # Reset en éxito
                
                time.sleep(0.05 if not self.degraded_mode else 0.1)
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                
                if consecutive_errors <= 3:
                    logging.error(f"Error en procesamiento ({consecutive_errors}): {e}")
                elif consecutive_errors >= max_consecutive_errors:
                    logging.error(f"Demasiados errores en procesamiento, pausando")
                    time.sleep(2)
                    consecutive_errors = 0
                
                time.sleep(0.1)
    
    def _export_seismic_event(self, result_data, window_data):
        """Exporta evento sísmico detectado en JSON y MiniSEED"""
        try:
            timestamp = datetime.now()
            event_id = timestamp.strftime("monitor_event_%Y%m%d_%H%M%S")
            
            # Datos del evento
            event_info = {
                'event_id': event_id,
                'timestamp': timestamp.isoformat(),
                'event_type': result_data.get('event_type', 'SISMO_CONFIRMADO'),
                'station_info': {
                    'station_id': 'SKYALERT_001',
                    'station_name': 'SKYALERT AI SEISMIC STATION 001',
                    'latitude': 19.4326,
                    'longitude': -99.1332,
                    'elevation': 2240.0,
                    'location': 'Ciudad de México, México'
                },
                'detection': result_data['result'][0],
                'magnitude_local': result_data['result'][1],
                'raw_output': result_data['result'][2],
                'processing_time': result_data['processing_time'],
                'processing_id': result_data['processing_id'],
                'system_info': {
                    'total_detections': self.detection_count,
                    'total_windows': self.processing_count,
                    'uptime_hours': (time.time() - self.start_time) / 3600
                }
            }
            
            # Exportar JSON
            json_path = f"events_monitor/{event_id}.json"
            import json
            with open(json_path, 'w') as f:
                json.dump(event_info, f, indent=2)
            
            logging.critical(
                f"🚨 SISMO CONFIRMADO EXPORTADO 🚨\n"
                f"ID: {event_id}\n"
                f"Magnitud Local: {result_data['result'][1]:.1f if result_data['result'][1] else 'N/A'}\n"
                f"Raw Output: {result_data['result'][2]:.1f}\n"
                f"Archivo: {json_path}"
            )
            
        except Exception as e:
            logging.error(f"Error exportando evento sísmico: {e}")
    
    def _ultra_fast_processing(self):
        """Procesamiento ultra-rápido - LÓGICA EXACTA DEL SIMULADOR"""
        current_time = time.time()
        
        if current_time - self.last_processing_time < self.latency_target:
            return None
        
        window_data = self.buffer.get_latest_window()
        if window_data is None:
            return None
        
        processing_id = self.processing_count
        if self.processing_pipeline.submit_window(window_data, processing_id):
            self.processing_count += 1
            self.last_processing_time = current_time
            
            result_data = self.processing_pipeline.get_result(timeout=0.3)
            
            if result_data:
                processing_time = time.time() - current_time
                
                return {
                    'timestamp': datetime.now(),
                    'detection': result_data['result'][0],
                    'magnitude': result_data['result'][1],
                    'confidence': result_data['result'][2],
                    'processing_id': processing_id,
                    'processing_time': processing_time,
                    'window_data': window_data
                }
        
        return None
    
    def stop_monitoring(self):
        """Detiene el sistema de forma ordenada"""
        logging.info("Iniciando parada ordenada del sistema...")
        
        self.running = False
        
        # Detener workers de procesamiento
        try:
            self.processing_pipeline.stop_workers()
            logging.info("Workers de procesamiento detenidos")
        except Exception as e:
            logging.error(f"Error deteniendo workers: {e}")
        
        # Cerrar conexión
        try:
            self.connector.close()
            logging.info("Conexión AnyShake cerrada")
        except Exception as e:
            logging.error(f"Error cerrando conexión: {e}")
        
        # Esperar threads con timeout
        threads_to_wait = []
        if self.data_thread and self.data_thread.is_alive():
            threads_to_wait.append((self.data_thread, "DataAcquisition"))
        if self.processing_thread and self.processing_thread.is_alive():
            threads_to_wait.append((self.processing_thread, "Processing"))
        if self.health_thread and self.health_thread.is_alive():
            threads_to_wait.append((self.health_thread, "HealthMonitor"))
        
        for thread, name in threads_to_wait:
            try:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logging.warning(f"Thread {name} no terminó en 5 segundos")
                else:
                    logging.info(f"Thread {name} terminado correctamente")
            except Exception as e:
                logging.error(f"Error esperando thread {name}: {e}")
        
        # Estadísticas finales
        uptime = time.time() - self.start_time
        logging.info(
            f"✅ Sistema detenido correctamente\n"
            f"Tiempo de operación: {uptime/3600:.1f} horas\n"
            f"Ventanas procesadas: {self.processing_count}\n"
            f"Detecciones: {self.detection_count}\n"
            f"Paquetes TCP: {self.connector.packet_count}\n"
            f"Reconexiones: {self.connector.total_reconnections}"
        )

def main():
    import argparse
    import signal
    
    # Configurar logging robusto para operación 24/7
    from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
    
    # Log principal con rotación por tamaño
    log_handler = RotatingFileHandler(
        'logs/creime_rt_realtime.log', 
        maxBytes=50*1024*1024,  # 50MB por archivo
        backupCount=10  # Mantener 10 archivos (500MB total)
    )
    log_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    
    # Log diario para análisis histórico
    daily_handler = TimedRotatingFileHandler(
        'logs/creime_rt_daily.log',
        when='midnight',
        interval=1,
        backupCount=30  # 30 días de historial
    )
    daily_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    
    # Configurar logger root
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    logger.addHandler(daily_handler)
    
    # Crear directorio de logs si no existe
    os.makedirs('logs', exist_ok=True)
    
    parser = argparse.ArgumentParser(description='CREIME_RT Tiempo Real - Sistema 24/7')
    parser.add_argument('--host', default='localhost', help='Host de AnyShake Observer')
    parser.add_argument('--port', type=int, default=30000, help='Puerto de AnyShake Observer')
    parser.add_argument('--model_path', help='Ruta del modelo CREIME_RT')
    parser.add_argument('--no-gui', action='store_true', help='Ejecutar sin interfaz gráfica (modo servidor)')
    
    args = parser.parse_args()
    
    # Configurar manejo de señales para parada ordenada
    monitor = None
    
    def signal_handler(signum, frame):
        logging.info(f"Señal {signum} recibida, iniciando parada ordenada...")
        if monitor:
            monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        monitor = CreimeRTRealTimeMonitor(args.host, args.port, args.model_path, no_gui=args.no_gui)
        logging.info("=== INICIANDO CREIME_RT MONITOR 24/7 ===")
        monitor.start_monitoring()
    except Exception as e:
        logging.critical(f"Error crítico en main: {e}")
        if monitor:
            monitor.stop_monitoring()
        sys.exit(1)

if __name__ == "__main__":
    # Configurar para operación 24/7
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Optimizaciones de sistema
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    
    main()