// FILE: lib/live_detection_screen.dart

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

import 'providers.dart';
import 'services/base_detector.dart';
import 'services/yolo_detector.dart';
import 'widgets/bounding_box_painter.dart';

// NOTE: google_mlkit_commons is still needed for ML Kit path detectors.
// When YOLO is active, _buildInputImage() is bypassed entirely.

class LiveDetectionScreen extends ConsumerStatefulWidget {
  const LiveDetectionScreen({super.key});

  @override
  ConsumerState<LiveDetectionScreen> createState() =>
      _LiveDetectionScreenState();
}

class _LiveDetectionScreenState extends ConsumerState<LiveDetectionScreen> {
  CameraController? _cameraController;

  List<DetectionResult> _results = [];
  Size _imageSize = Size.zero;
  bool _isProcessing = false;
  bool _isCameraReady = false;
  bool _isFrontCamera = false;
  int _fps = 0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final camera = cameras.first;
    _isFrontCamera = camera.lensDirection == CameraLensDirection.front;

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21, // NV21 works for both paths
    );

    await _cameraController!.initialize();
    if (!mounted) return;

    setState(() => _isCameraReady = true);
    _cameraController!.startImageStream(_onCameraFrame);
  }

  Future<void> _onCameraFrame(CameraImage image) async {
    if (_isProcessing) return;
 
    // ref.read is from riverpod i guess, it allows us to read the current active detector and distance estimator
      // it needs a provider to read from, and it will return the current value of that provider'
      // A provider is similar to react's context, it holds the state of the app and allows us to read and write that state from anywhere in the app
    final detectorState = ref.read(activeDetectorProvider);
  if (detectorState is! AsyncData) return; // Skip frame if still loading/switching
  final detector = detectorState.value; // Now we are 100% sure it's initialized
     _isProcessing = true;

    try {
      
      

      final distanceSvc = ref.read(activeDistanceProvider);
      final frameSize   = Size(image.width.toDouble(), image.height.toDouble());

      List<Detection> detections;

      if (detector is YoloDetector) {
        // tflite_flutter path — pass raw NV21 bytes directly
        detections = await _runYolo(detector, image);
        print('YOLO detected ${detections.length} objects');
      } else {
        // ML Kit path (InputImage-based detectors)
        final inputImage = _buildInputImage(image);
        if (inputImage == null) return;
        detections = await detector.detect(inputImage);
      }

      if (!mounted) return;

      final results = detections.map((d) {
        return DetectionResult(
          boundingBox: d.boundingBox,
          label:       d.label,
          confidence:  d.confidence,
          distance:    distanceSvc.estimate(d.boundingBox, frameSize),
        );
      }).toList();

      _frameCount++;
      final now = DateTime.now();
      if (now.difference(_lastFpsUpdate).inSeconds >= 1) {
        _fps        = _frameCount;
        _frameCount = 0;
        _lastFpsUpdate = now;
      }

      setState(() {
        _results   = results;
        _imageSize = frameSize;
      });
    } finally {
      _isProcessing = false;
    }
  }

  /// Feeds a [CameraImage] to the YoloDetector.
  /// tflite_flutter works on raw NV21 bytes — no InputImage wrapper needed.
  Future<List<Detection>> _runYolo(
      YoloDetector yolo, CameraImage image) async {
    // NV21 stores Y plane first, then interleaved VU.
    // Concatenate all planes into one contiguous buffer.
    final allBytes = BytesBuilder();
    for (final plane in image.planes) {
      allBytes.add(plane.bytes);
    }

    print("running YOLO");

    return yolo.detectFromBytes(
      bytes:  allBytes.toBytes(),
      imageW: image.width,
      imageH: image.height,
      rotation: _cameraController!.description.sensorOrientation,
    );
  }

  /// Builds an [InputImage] for ML Kit detectors.
  InputImage? _buildInputImage(CameraImage image) {
    final camera          = _cameraController!.description;
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (camera.lensDirection == CameraLensDirection.front) {
      rotation = InputImageRotationValue.fromRawValue(360 - sensorOrientation);
    } else {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    }
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null;
    if (format != InputImageFormat.nv21 &&
        format != InputImageFormat.bgra8888) return null;

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size:        Size(image.width.toDouble(), image.height.toDouble()),
        rotation:    rotation,
        format:      format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    super.dispose();
  }

  // ─── UI ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final detectorType = ref.watch(activeDetectorTypeProvider);
    final distanceType = ref.watch(activeDistanceTypeProvider);

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_isCameraReady && _cameraController != null)
            SizedBox.expand(child: CameraPreview(_cameraController!))
          else
            const Center(
              child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
            ),

          if (_isCameraReady && _imageSize != Size.zero)
            LayoutBuilder(builder: (ctx, constraints) {
              final widgetSize =
                  Size(constraints.maxWidth, constraints.maxHeight);
              return CustomPaint(
                painter: BoundingBoxPainter(
                  results:     _results,
                  imageSize:   _imageSize,
                  widgetSize:  widgetSize,
                  isFrontCamera: _isFrontCamera,
                ),
              );
            }),

          SafeArea(child: _buildTopBar(detectorType)),

          Positioned(
            bottom: 0, left: 0, right: 0,
            child: _buildBottomStats(),
          ),
        ],
      ),
    );
  }

  Widget _buildTopBar(DetectorType detectorType) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          _GlassButton(
            icon:  Icons.arrow_back_ios_new_rounded,
            onTap: () => Navigator.pop(context),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'LIVE DETECTION',
              style: TextStyle(
                color:       Colors.white,
                fontSize:    13,
                fontWeight:  FontWeight.w700,
                letterSpacing: 2.5,
              ),
            ),
          ),
          _StatusBadge(label: detectorType.label, color: const Color(0xFF00E5FF)),
          const SizedBox(width: 6),
          _StatusBadge(label: '$_fps fps',          color: const Color(0xFF7C4DFF)),
          const SizedBox(width: 6),
          _StatusBadge(label: '${_results.length} obj', color: const Color(0xFF00BFA5)),
        ],
      ),
    );
  }

  Widget _buildBottomStats() {
    if (_results.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin:  Alignment.bottomCenter,
            end:    Alignment.topCenter,
            colors: [Colors.black.withOpacity(0.8), Colors.transparent],
          ),
        ),
        child: Center(
          child: Text(
            'Point camera at objects...',
            style: TextStyle(
              color:         Colors.white.withOpacity(0.4),
              fontSize:      13,
              letterSpacing: 1,
            ),
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin:  Alignment.bottomCenter,
          end:    Alignment.topCenter,
          colors: [Colors.black.withOpacity(0.9), Colors.transparent],
        ),
      ),
      child: Wrap(
        spacing:    8,
        runSpacing: 8,
        children: _results.take(5).map((r) {
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color:        Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border:       Border.all(
                  color: const Color(0xFF00E5FF).withOpacity(0.4)),
            ),
            child: Text(
              '${r.label} · ${(r.confidence * 100).toStringAsFixed(0)}% · ${r.distance}',
              style: const TextStyle(
                color:      Colors.white,
                fontSize:   11,
                fontWeight: FontWeight.w600,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// ─── Shared small widgets ────────────────────────────────────────────────────

class _GlassButton extends StatelessWidget {
  final IconData   icon;
  final VoidCallback onTap;
  const _GlassButton({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width:  40,
        height: 40,
        decoration: BoxDecoration(
          color:        Colors.white.withOpacity(0.15),
          borderRadius: BorderRadius.circular(10),
          border:       Border.all(color: Colors.white.withOpacity(0.2)),
        ),
        child: Icon(icon, color: Colors.white, size: 18),
      ),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final String label;
  final Color  color;
  const _StatusBadge({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color:        color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
        border:       Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(
        label,
        style: TextStyle(
          color:         color,
          fontSize:      11,
          fontWeight:    FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
