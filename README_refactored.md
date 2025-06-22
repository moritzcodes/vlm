# Liquid Handler Monitor - Refactored Architecture

## Overview

The Liquid Handler Monitor has been refactored from a monolithic script into a modular, maintainable architecture. This new structure separates concerns and makes the codebase much easier to understand, test, and extend.

## Architecture

### Core Components

#### 1. `config.py` - Configuration Management
- **Purpose**: Centralized configuration and constants
- **Contains**:
  - UI color schemes and styling
  - Camera settings and search indices
  - AI model configuration
  - Filter parameters and crop presets
  - Status color mappings

#### 2. `data_models.py` - Data Structures
- **Purpose**: Type-safe data models and definitions
- **Contains**:
  - `ErrorEvent` - Error tracking data
  - `ProcedureStep` - Procedure step definitions
  - `TrackedObject` - Object tracking data
  - `AnalysisResult` - AI analysis results
  - `FeedbackItem` - UI feedback items
  - `ProcedureDefinitions` - Procedure workflows

#### 3. `camera_manager.py` - Camera Operations
- **Purpose**: Handle all camera-related functionality
- **Features**:
  - Auto-detection of cameras (iPhone preferred)
  - Frame capture and management
  - Camera configuration and settings
  - Error handling and reconnection

#### 4. `ai_analyzer.py` - AI Analysis
- **Purpose**: Interface with Google Gemini 2.5 Pro for frame analysis
- **Features**:
  - Model initialization and setup
  - Context-aware prompt generation
  - Frame analysis with procedure context
  - Response parsing and structured data extraction

#### 5. `frame_processor.py` - Image Processing
- **Purpose**: Handle frame cropping and filtering
- **Features**:
  - 8 different image filters (grayscale, HSV, blur, edge, etc.)
  - Crop region management with presets
  - Undo/redo functionality for crop operations
  - Real-time parameter adjustment

#### 6. `ui_components.py` - User Interface
- **Purpose**: All UI rendering and overlay drawing
- **Features**:
  - Professional Iron Man-style interface
  - Status panels and indicators
  - VLM feedback system
  - Well status grid visualization
  - Crop control UI

#### 7. `liquid_handler_monitor_refactored.py` - Main Application
- **Purpose**: Orchestrate all components and handle application logic
- **Features**:
  - Component initialization and coordination
  - Main application loop
  - Event handling (keyboard, mouse)
  - Threading for AI analysis
  - State management

## File Structure

```
liquid_handler_monitor/
├── config.py                              # Configuration and constants
├── data_models.py                         # Data structures and models
├── camera_manager.py                      # Camera handling
├── ai_analyzer.py                         # AI analysis module
├── frame_processor.py                     # Image processing
├── ui_components.py                       # UI rendering
├── liquid_handler_monitor_refactored.py   # Main application
├── liquid_handler_monitor.py              # Original monolithic version
└── README_refactored.md                   # This documentation
```

## Benefits of Refactoring

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Changes to UI don't affect AI analysis logic
- Camera issues don't break the entire application

### 2. **Maintainability**
- Easy to locate and fix bugs in specific components
- Clear interfaces between modules
- Consistent error handling and logging

### 3. **Testability**
- Each component can be unit tested independently
- Mock objects can replace dependencies for testing
- Easier to write integration tests

### 4. **Extensibility**
- New features can be added to specific modules
- Easy to swap out implementations (e.g., different AI models)
- Plugin architecture possible for new filters or UI components

### 5. **Code Reusability**
- Components can be reused in other projects
- Common functionality is centralized
- Reduced code duplication

## Usage

### Running the Refactored Version

```bash
# Run the new modular version
python liquid_handler_monitor_refactored.py

# Or run the original version for comparison
python liquid_handler_monitor.py
```

### Key Controls

- **SPACE**: Pause/Resume monitoring
- **1**: Start Blue-Red Mixing procedure
- **2**: Start PCR Master-Mix procedure
- **C**: Toggle crop control UI
- **ESC**: Exit application
- **1-8**: Apply different filters
- **+/-**: Adjust filter parameters
- **R**: Reset crop region
- **U**: Undo last crop change

### Environment Setup

Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Component Interaction Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Main Monitor   │────│  Camera Manager  │────│  Frame Capture  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ├─────────────────────────────────────────────────────────┐
         │                                                         │
         ▼                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Frame Processor │────│   AI Analyzer    │────│   UI Renderer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Crop & Filters  │    │ Gemini Analysis  │    │ Iron Man UI     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Configuration

### Adding New Filters

In `frame_processor.py`, add to the `_apply_filter` method:

```python
elif self.current_filter == "new_filter":
    # Your filter implementation
    return processed_frame
```

### Adding New Procedures

In `data_models.py`, extend the `ProcedureDefinitions.get_procedures()` method:

```python
"new_procedure": [
    ProcedureStep("step1", "Step 1", "Description", ["blue"], ["A1"], (10, 30)),
    # ... more steps
]
```

### Customizing UI Colors

In `config.py`, modify the `UIColors` dataclass:

```python
@dataclass
class UIColors:
    background: Tuple[int, int, int] = (15, 20, 25)  # Dark blue-gray
    accent: Tuple[int, int, int] = (100, 200, 255)   # Arc reactor blue
    # ... other colors
```

## Performance Considerations

- **Threading**: AI analysis runs in background threads to maintain UI responsiveness
- **Frame Processing**: Efficient OpenCV operations for real-time performance
- **Memory Management**: Automatic cleanup of resources and limited error history
- **Caching**: Reuse of computed values where possible

## Future Enhancements

1. **Plugin System**: Load custom filters and analyzers dynamically
2. **Database Integration**: Store error events and analysis results
3. **Web Interface**: Remote monitoring capabilities
4. **Multiple Cameras**: Support for multiple camera feeds
5. **Advanced Analytics**: Historical trend analysis and reporting

## Migration from Original

The refactored version maintains full compatibility with the original functionality while providing a cleaner architecture. Both versions can coexist during the transition period.

### Key Differences

- **Modular Structure**: Code split into logical components
- **Better Error Handling**: Centralized error management
- **Improved Threading**: Background AI analysis with proper synchronization
- **Enhanced UI**: More responsive and maintainable interface code
- **Configuration Management**: Centralized settings and easy customization

## Dependencies

- `opencv-python`: Computer vision operations
- `google-genai`: AI analysis with Gemini 2.5 Pro
- `Pillow`: Image processing
- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations

## Troubleshooting

### Common Issues

1. **Camera Not Found**: Check camera connections and permissions
2. **AI Analysis Errors**: Verify Google API key in `.env` file
3. **Performance Issues**: Reduce frame rate or disable expensive filters
4. **UI Responsiveness**: Ensure AI analysis is running in background threads

### Debug Mode

Enable detailed logging by modifying the logging level in the main monitor:

```python
logging.basicConfig(level=logging.DEBUG)
```

This refactored architecture provides a solid foundation for future development while maintaining all the existing functionality in a much more maintainable form. 