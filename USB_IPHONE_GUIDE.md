# 🔌 USB iPhone Camera Connection Guide

## Why Use USB Connection?

USB connection provides several advantages over wireless Continuity Camera:

✅ **More Stable Connection** - No wireless interference or timeout issues  
✅ **Better Performance** - Lower latency and more consistent frame rates  
✅ **No Battery Drain** - iPhone charges while connected  
✅ **Reliable for Long Sessions** - Perfect for extended VLM processing  

## Setup Instructions

### 1. Connect Your iPhone
1. Use a **Lightning to USB-C** or **Lightning to USB-A** cable
2. Connect iPhone to your Mac
3. If prompted, **trust this computer** on your iPhone
4. Keep your iPhone unlocked during camera usage

### 2. Enable Camera Access
1. On your iPhone: Settings > Privacy & Security > Camera
2. Make sure camera access is enabled for system services
3. You may need to enable "Continuity Camera" in System Preferences on Mac

### 3. Test the Connection
```bash
# Check if iPhone camera is detected
python camera_switcher.py list

# Test USB connection stability
python camera_switcher.py test 0

# Switch to iPhone camera
python camera_switcher.py iphone
```

## USB Optimizations Enabled

The system automatically detects and applies USB iPhone optimizations:

🔌 **USB Connection Timeout**: 30 seconds  
🔄 **Auto-Reconnection**: Up to 3 attempts  
⏱️ **Frame Read Timeout**: 5 seconds  
🛡️ **Connection Monitoring**: Automatic failure detection  
📱 **Keep-Alive**: Continuous frame reading to prevent timeout  

## Troubleshooting

### Camera Disconnects
- **Ensure iPhone stays unlocked** during camera usage
- **Check cable connection** - try a different cable if issues persist
- **Restart both devices** if connection becomes unstable

### Low Frame Rates
- **Close other camera apps** on iPhone and Mac
- **Ensure good lighting** for optimal camera performance
- **Check iPhone storage** - low storage can affect camera performance

### Connection Errors
```bash
# If camera isn't detected, try:
python camera_switcher.py detect

# Force reconnection:
python camera_switcher.py iphone

# Test specific camera index:
python camera_switcher.py test 0
```

## Recommended Workflow

1. **Connect iPhone via USB**
2. **Run camera detection**: `python camera_switcher.py list`
3. **Switch to iPhone**: `python camera_switcher.py iphone`
4. **Launch VLM system**: `python quick_launch.py`

## Performance Comparison

| Connection Type | Stability | Latency | Battery | Setup |
|----------------|-----------|---------|---------|-------|
| **USB Cable** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Wireless | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Use USB connection for production VLM processing and wireless for quick testing. 