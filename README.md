# CREIME_RT Monitor - Sistema de Alerta Sísmica Temprana

## 🌍 Descripción General

**CREIME_RT Monitor** es un sistema avanzado de alerta sísmica temprana diseñado para operación continua 24/7. Utiliza el modelo de Deep Learning **CREIME_RT** (Convolutional Recurrent Model for Earthquake Identification and Magnitude Estimation - Real Time) para detectar y estimar la magnitud de eventos sísmicos en tiempo real con latencia mínima.

El sistema está optimizado para ejecutarse en **NVIDIA Jetson Orin Nano** y procesa datos sísmicos de tres componentes provenientes de sensores **AnyShake**, proporcionando alertas sísmicas con una latencia promedio de **1.5 segundos**.

## 🎯 Propósito y Diseño

### Objetivo Principal
Proporcionar **detección sísmica en tiempo real** con:
- **Latencia ultra-baja**: ~1.5 segundos desde el evento hasta la alerta
- **Alta precisión**: >99.8% en diferenciación ruido/sismo
- **Operación continua**: Diseñado para funcionar 24/7 sin intervención
- **Estimación de magnitud**: RMSE de 0.38 unidades

### Casos de Uso
- **Sistemas de alerta temprana** para poblaciones en riesgo sísmico
- **Monitoreo sísmico continuo** en tiempo real
- **Investigación sismológica** con procesamiento automatizado
- **Integración con redes de sensores** distribuidos

## 🧠 ¿Qué es CREIME_RT?

**CREIME_RT** es un modelo de Deep Learning desarrollado por el equipo SAIPy que combina arquitecturas **Convolucionales** y **Recurrentes** para:

### Funcionamiento del Modelo
1. **Entrada**: Ventanas de 30 segundos (3000 muestras) de datos sísmicos de 3 componentes
2. **Procesamiento**: Red neuronal que analiza patrones sísmicos vs ruido
3. **Salida**: Valor numérico que indica:
   - **-4.0**: Ruido (pre-señal)
   - **> -0.5**: Evento sísmico detectado
   - **> 0.0**: Magnitud estimada del evento

### Capacidades del Modelo
- **Precisión**: >99.80% en clasificación ruido/terremoto
- **Estimación de magnitud**: RMSE de 0.38 unidades
- **Rango de detección**: Magnitudes desde 1.0 hasta 7.0+
- **Tiempo de inferencia**: ~300ms en Jetson Orin Nano

## 🔧 Núcleo del Sistema

### Arquitectura Principal

```
AnyShake Sensor → TCP Stream → Buffer Circular → CREIME_RT → Detección → Alerta
                                      ↓
                              Visualización en Tiempo Real
```

### Componentes Clave

#### 1. **UltraFastBuffer**
- **Buffer circular** de 30 segundos (3000 muestras)
- **Ventana deslizante** con traslape del 96.67%
- **Actualización**: Cada 1 segundo con datos nuevos
- **Sincronización**: Perfecta con llegada de datos AnyShake

#### 2. **OptimizedHybridFilter**
Pipeline de procesamiento de señales:
```
Datos Crudos → Normalización Z-Score → Filtro 1-45Hz → Conversión Gals → CREIME_RT
```

#### 3. **UltraFastProcessingPipeline**
- **Workers multihilo** para procesamiento paralelo
- **Queue system** para manejo eficiente de ventanas
- **GPU acceleration** en Jetson Orin Nano
- **Recuperación automática** de errores

#### 4. **RealTimeVisualizer**
- **Visualización en tiempo real** de las 3 componentes sísmicas
- **Escalado dinámico** basado en amplitud real
- **Marcadores CREIME_RT** para ventanas de procesamiento
- **Información del sistema** en tiempo real

### Flujo de Procesamiento

1. **Adquisición**: AnyShake envía paquetes cada 1000ms (100 muestras)
2. **Filtrado**: Normalización Z-Score + filtro pasa-banda 1-45Hz
3. **Buffering**: Ventana deslizante de 30 segundos actualizada cada segundo
4. **Inferencia**: CREIME_RT procesa ventana de 3000×3 muestras
5. **Detección**: Umbral -0.5 para clasificación evento/ruido
6. **Alerta**: Confirmación inmediata (1 ventana consecutiva)

## ⚡ Análisis de Latencia

### Componentes de Latencia
- **Llegada de datos AnyShake**: 0-1000ms (promedio 500ms)
- **Procesamiento de buffer**: ~1ms
- **Inferencia CREIME_RT**: ~300ms
- **Detección y alerta**: ~1ms

### **Latencia Total: ~1.5 segundos**

### Comparación con Sistema HomeSeismo HS301
- **HS301**: de 1.5 a 5 segundos de latencia

## 🛡️ Estabilidad y Confiabilidad

### Características de Estabilidad 24/7

#### Gestión de Memoria
- **Buffers circulares** con límites fijos
- **Limpieza automática** cada hora
- **Garbage collection** periódico
- **Monitoreo de uso** de memoria

#### Recuperación de Errores
- **Reconexión automática** a AnyShake
- **Manejo de excepciones** en todos los bucles críticos
- **Recuperación de workers** CREIME_RT
- **Logs detallados** para diagnóstico

#### Tolerancia a Fallos
- **Threads daemon** para cierre seguro
- **Timeouts configurables** en todas las operaciones
- **Parada ordenada** de componentes
- **Modo degradado** sin visualización

### Métricas de Confiabilidad
- **Tiempo de actividad esperado**: >99.5%
- **Recuperación de fallos**: <3 segundos
- **Uso de memoria**: Estable con limpieza periódica
- **Latencia**: Consistente (~1.5s ±0.2s)

## 🚀 Instalación y Uso

### Requisitos del Sistema
- **Hardware**: NVIDIA Jetson Orin Nano (recomendado)
- **OS**: Ubuntu 20.04+ con soporte CUDA
- **Python**: 3.8+
- **Memoria**: 8GB RAM mínimo
- **Almacenamiento**: 32GB disponibles

### Dependencias
```bash
pip install numpy scipy matplotlib obspy psutil
pip install saipy  # Modelo CREIME_RT
```

### Ejecución
```bash
python CREIME_RT_Monitor.py --host localhost --port 30000
```

### Parámetros de Configuración
- `--model_path`: Ruta del modelo CREIME_RT (default: ../saipy/saved_models/)
- `--host`: Host de AnyShake Observer (default: localhost)
- `--port`: Puerto de AnyShake Observer (default: 30000)

### Detener el Sistema
- **Ctrl+C** en consola
- **Cerrar ventana** del visualizador

## 📊 Salidas del Sistema

### Logs en Tiempo Real
```
[2025-01-XX 12:34:56] CREIME_RT Raw Output: -3.85
[2025-01-XX 12:34:57] DETECCIÓN: Confianza 0.45 > -0.5
[2025-01-XX 12:34:57] 🚨 SISMO CONFIRMADO 🚨
```

### Archivos Generados
- **Logs**: `logs/creime_rt_monitor.log`
- **Eventos JSON (sólo cuando confirma un sismo)**: `events_monitor/monitor_event_YYYYMMDD_HHMMSS.json`
- **Datos MiniSEED (sólo cuando confirma un sismo)**: `events_monitor/monitor_event_YYYYMMDD_HHMMSS.mseed`

### Visualización
- **Gráficos en tiempo real** de las 3 componentes sísmicas
- **Escalado dinámico** basado en amplitud real
- **Información del sistema** (paquetes, detecciones, ventanas)
- **Marcadores de ventana** CREIME_RT

## 🔬 Especificaciones Técnicas

### Procesamiento de Señales
- **Frecuencia de muestreo**: 100 Hz
- **Componentes**: ENZ (vertical), ENE (este-oeste), ENN (norte-sur)
- **Filtrado**: Pasa-banda 1-45 Hz
- **Normalización**: Z-Score en ventana móvil
- **Conversión**: mg → Gals (factor 0.119)

### Parámetros CREIME_RT
- **Ventana de análisis**: 30 segundos (3000 muestras)
- **Umbral de detección**: -0.5
- **Umbral de magnitud**: 0.0
- **Ventanas consecutivas**: 1 (confirmación inmediata)
- **Formato de entrada**: [1, 3000, 3] float32

### Rendimiento
- **Tasa de procesamiento**: 1 ventana/segundo
- **Throughput**: ~300 muestras/segundo
- **Uso de CPU**: ~30% en Jetson Orin Nano
- **Uso de GPU**: ~20% durante inferencia
- **Memoria RAM**: ~2GB estable

## 📁 Estructura del Proyecto

```
SAIPy/
├── CREIME_RT_Monitor.py          # Sistema principal
├── README.md                     # Este archivo
├── logs/                         # Logs del sistema
│   └── creime_rt_monitor.log
├── events/               # Eventos detectados
│   ├── monitor_event_*.json
│   └── monitor_event_*.mseed
└── requirements.txt              # Dependencias
```

## 🔍 Monitoreo y Diagnóstico

### Indicadores de Salud del Sistema
- **Paquetes recibidos**: Rate de paquetes AnyShake
- **Ventanas procesadas**: Throughput de CREIME_RT
- **Detecciones**: Eventos identificados
- **Latencia**: Tiempo de procesamiento por ventana

### Logs de Diagnóstico
- **INFO**: Operación normal del sistema
- **WARNING**: Situaciones recuperables
- **ERROR**: Errores que requieren atención
- **CRITICAL**: Alertas sísmicas confirmadas

## 🚨 Limitaciones y Consideraciones

### Limitaciones Técnicas
- **Dependencia de AnyShake**: Punto único de falla
- **Latencia inherente**: Mínimo 1 segundo por parseo de datos AnyShake
- **Magnitudes altas**: Subestimación en Ml ≥ 5.5
- **Ruido local**: Puede generar falsos positivos si la instalación del sensor es incorrecta

### Consideraciones de Despliegue
- **Conectividad estable** requerida con AnyShake
- **Alimentación ininterrumpida** para operación 24/7
- **Monitoreo externo** recomendado para alta disponibilidad
- **Backup de datos** para eventos críticos

## 📄 Licencias

Este software es propiedad exclusiva de **SkyAlert de México S.A. de C.V.** Todos los derechos están reservados. El uso, copia, modificación, distribución o reproducción total o parcial de este desarrollo está estrictamente limitado al uso interno autorizado por SkyAlert de México S.A. de C.V. Queda prohibido su uso por terceros sin el consentimiento expreso y por escrito de la empresa.

**© 2025 SkyAlert de México S.A. de C.V. — Todos los derechos reservados.**

##  Agradecimientos

- **SAIPy Team**: Por desarrollar y entrenar el modelo CREIME_RT
- **AnyShake Project**: Por la infraestructura de adquisición de datos  
- **NVIDIA**: Por la plataforma Jetson Orin Nano

---

**Desarrollado por Ing. Isaac Pérez de SkyAlert de México S.A. de C.V.**  
*Sistema de Alerta Sísmica Temprana - Tecnología de Vanguardia*
