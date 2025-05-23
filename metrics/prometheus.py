import time
import os
import psutil
import platform
from threading import Lock
import gc
import logging
import folder_paths
# Import GPU info class
from ..core.gpu_info import CGPUInfo

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Singleton metrics registry
class MetricsRegistry:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsRegistry, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        # Initialize metrics data
        self.metrics = {}
        self.counters = {}
        self.gauges = {}
        self.start_time = time.time()

        # Initialize GPU info
        self.gpu_info = CGPUInfo()

        # Initialize basic counters and gauges
        self.add_gauge("comfyui_uptime_seconds", "ComfyUI server uptime in seconds")
        self.add_gauge("comfyui_memory_usage_bytes", "ComfyUI memory usage in bytes")
        self.add_gauge("comfyui_memory_free_bytes", "System memory free in bytes")
        self.add_gauge("comfyui_memory_total_bytes", "System total memory in bytes")
        self.add_gauge("comfyui_memory_usage_percent", "System memory usage percentage")
        self.add_gauge("comfyui_cpu_usage_percent", "ComfyUI CPU usage percentage")
        self.add_gauge("comfyui_disk_usage_bytes", "ComfyUI disk usage in bytes")
        self.add_gauge("comfyui_disk_free_bytes", "Disk free space in bytes")
        self.add_gauge("comfyui_disk_total_bytes", "Disk total space in bytes")
        self.add_gauge("comfyui_disk_usage_percent", "ComfyUI disk usage percentage")

        # Add PyTorch version info if available
        if TORCH_AVAILABLE:
            self.add_gauge("comfyui_pytorch_info", "PyTorch version information", labels={"version": torch.__version__})

        # Initialize GPU metrics
        self.add_gauge("comfyui_gpu_utilization_percent", "GPU utilization percentage")
        self.add_gauge("comfyui_gpu_temperature_celsius", "GPU temperature in Celsius")
        self.add_gauge("comfyui_gpu_vram_used_bytes", "GPU VRAM used in bytes")
        self.add_gauge("comfyui_gpu_vram_total_bytes", "GPU VRAM total in bytes")
        self.add_gauge("comfyui_gpu_vram_used_percent", "GPU VRAM usage percentage")

    def add_counter(self, name, help_text, labels=None):
        """Add a new counter metric"""
        if name not in self.counters:
            self.counters[name] = {
                "help": help_text,
                "type": "counter",
                "value": 0,
                "labels": labels or {}
            }

    def add_gauge(self, name, help_text, labels=None):
        """Add a new gauge metric"""
        if name not in self.gauges:
            self.gauges[name] = {
                "help": help_text,
                "type": "gauge",
                "value": 0,
                "labels": labels or {}
            }

    def increment_counter(self, name, value=1, labels=None):
        """Increment a counter metric"""
        if name in self.counters:
            self.counters[name]["value"] += value
            if labels:
                # Handle labeled metrics
                label_key = self._labels_to_key(labels)
                if "labeled_values" not in self.counters[name]:
                    self.counters[name]["labeled_values"] = {}
                if label_key not in self.counters[name]["labeled_values"]:
                    self.counters[name]["labeled_values"][label_key] = 0
                self.counters[name]["labeled_values"][label_key] += value

    def set_gauge(self, name, value, labels=None):
        """Set a gauge metric value"""
        if name in self.gauges:
            self.gauges[name]["value"] = value
            if labels:
                # Handle labeled metrics
                label_key = self._labels_to_key(labels)
                if "labeled_values" not in self.gauges[name]:
                    self.gauges[name]["labeled_values"] = {}
                self.gauges[name]["labeled_values"][label_key] = value

    def _labels_to_key(self, labels):
        """Convert labels dict to a unique string key"""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _format_labels(self, labels):
        """Format labels for Prometheus output"""
        if not labels:
            return ""
        label_strings = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_strings) + "}"

    def update_system_metrics(self):
        """Update system-related metrics"""
        # Update uptime
        self.set_gauge("comfyui_uptime_seconds", time.time() - self.start_time)

        # Set PyTorch version info if available
        if TORCH_AVAILABLE:
            self.set_gauge("comfyui_pytorch_info", 1, labels={"version": torch.__version__})

        # Update memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.set_gauge("comfyui_memory_usage_bytes", memory_info.rss)

        # Update system memory metrics
        system_memory = psutil.virtual_memory()
        self.set_gauge("comfyui_memory_free_bytes", system_memory.available)
        self.set_gauge("comfyui_memory_total_bytes", system_memory.total)
        self.set_gauge("comfyui_memory_usage_percent", system_memory.percent)

        # Update CPU usage
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            self.set_gauge("comfyui_cpu_usage_percent", cpu_percent)
        except:
            pass

        # Update disk usage for output directory
        try:
            output_dir = folder_paths.get_output_directory()
            disk_usage = psutil.disk_usage(output_dir)
            self.set_gauge("comfyui_disk_usage_bytes", disk_usage.used)
            self.set_gauge("comfyui_disk_free_bytes", disk_usage.free)
            self.set_gauge("comfyui_disk_total_bytes", disk_usage.total)
            self.set_gauge("comfyui_disk_usage_percent", disk_usage.percent)
        except:
            pass

        # Update GPU metrics
        try:
            gpu_status = self.gpu_info.getStatus()

            # Only proceed if we're not using CPU
            if gpu_status['device_type'] != 'cpu':
                # Handle multiple GPUs if present
                for gpu_index, gpu_data in enumerate(gpu_status['gpus']):
                    # Define labels for this GPU
                    gpu_labels = {"gpu_index": str(gpu_index)}

                    # GPU utilization
                    if gpu_data['gpu_utilization'] >= 0:
                        self.set_gauge("comfyui_gpu_utilization_percent",
                                      gpu_data['gpu_utilization'],
                                      labels=gpu_labels)

                    # GPU temperature
                    if gpu_data['gpu_temperature'] >= 0:
                        self.set_gauge("comfyui_gpu_temperature_celsius",
                                      gpu_data['gpu_temperature'],
                                      labels=gpu_labels)

                    # VRAM metrics
                    if gpu_data['vram_used'] >= 0 and gpu_data['vram_total'] >= 0:
                        self.set_gauge("comfyui_gpu_vram_used_bytes",
                                      gpu_data['vram_used'],
                                      labels=gpu_labels)
                        self.set_gauge("comfyui_gpu_vram_total_bytes",
                                      gpu_data['vram_total'],
                                      labels=gpu_labels)

                    # VRAM percentage
                    if gpu_data['vram_used_percent'] >= 0:
                        self.set_gauge("comfyui_gpu_vram_used_percent",
                                      gpu_data['vram_used_percent'],
                                      labels=gpu_labels)
        except Exception as e:
            logging.error(f"Error updating GPU metrics: {str(e)}")

    def format_prometheus(self):
        """Format all metrics for Prometheus scraping"""
        lines = []

        # Format counters
        for name, counter in self.counters.items():
            lines.append(f"# HELP {name} {counter['help']}")
            lines.append(f"# TYPE {name} {counter['type']}")
            lines.append(f"{name} {counter['value']}")

            # Add labeled values if any
            if "labeled_values" in counter:
                for label_key, value in counter["labeled_values"].items():
                    label_dict = {k: v for k, v in [item.split("=") for item in label_key.split("|")]}
                    formatted_labels = self._format_labels(label_dict)
                    lines.append(f"{name}{formatted_labels} {value}")

        # Format gauges
        for name, gauge in self.gauges.items():
            lines.append(f"# HELP {name} {gauge['help']}")
            lines.append(f"# TYPE {name} {gauge['type']}")
            lines.append(f"{name} {gauge['value']}")

            # Add labeled values if any
            if "labeled_values" in gauge:
                for label_key, value in gauge["labeled_values"].items():
                    label_dict = {k: v for k, v in [item.split("=") for item in label_key.split("|")]}
                    formatted_labels = self._format_labels(label_dict)
                    lines.append(f"{name}{formatted_labels} {value}")

        return "\n".join(lines)

# Create a global registry instance
registry = MetricsRegistry()

def get_metrics():
    """Get formatted Prometheus metrics"""
    try:
        # Update system metrics before returning
        registry.update_system_metrics()
        return registry.format_prometheus()
    except Exception as e:
        logging.error(f"Error generating metrics: {str(e)}")
        return "# Error generating metrics"