// FILE: lib/live_detection_screen.dart

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

import 'services/detection_service.dart';
import 'services/distance_service.dart';
import 'widgets/bounding_box_painter.dart';

class LiveDetectionScreen extends StatefulWidget {
  const LiveDetectionScreen({super.key});

  @override
  State<LiveDetectionScreen> createState() => _LiveDetectionScreenState();
}

class _LiveDetectionScreenState extends State<LiveDetectionScreen> {
  CameraController? _cameraController;
  final _detectionService = DetectionService();
  final _distanceService = DistanceService();

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
    _detectionService.initialize();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final camera = cameras.first; // back camera
    _isFrontCamera = camera.lensDirection == CameraLensDirection.front;

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,       // balance between quality & speed
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21, // Android; iOS uses bgra8888
    );

    await _cameraController!.initialize();
    if (!mounted) return;

    setState(() => _isCameraReady = true);

    _cameraController!.startImageStream(_onCameraFrame);
  }

  /// Called for every camera frame.
  Future<void> _onCameraFrame(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final inputImage = _buildInputImage(image);
      if (inputImage == null) return;

      final objects = await _detectionService.processInputImage(inputImage);

      if (!mounted) return;

      final results = objects.map((obj) {
        final label = labelFor(obj);
        final confidence = confidenceFor(obj);
        final distance = _distanceService.estimate(
          obj.boundingBox,
          Size(image.width.toDouble(), image.height.toDouble()),
        );
        return DetectionResult(
          boundingBox: obj.boundingBox,
          label: label,
          confidence: confidence,
          distance: distance,
        );
      }).toList();

      // FPS counter
      _frameCount++;
      final now = DateTime.now();
      if (now.difference(_lastFpsUpdate).inSeconds >= 1) {
        _fps = _frameCount;
        _frameCount = 0;
        _lastFpsUpdate = now;
      }

      setState(() {
        _results = results;
        _imageSize = Size(image.width.toDouble(), image.height.toDouble());
      });
    } finally {
      _isProcessing = false;
    }
  }

  InputImage? _buildInputImage(CameraImage image) {
    final camera = _cameraController!.description;

    // Rotation mapping
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

    // Only NV21 / BGRA8888 are supported by ML Kit
    if (format != InputImageFormat.nv21 && format != InputImageFormat.bgra8888) {
      return null;
    }

    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _detectionService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Camera preview ─────────────────────────────────────────────
          if (_isCameraReady && _cameraController != null)
            _buildCameraPreview()
          else
            const Center(
              child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
            ),

          // ── Bounding boxes ─────────────────────────────────────────────
          if (_isCameraReady && _imageSize != Size.zero)
            LayoutBuilder(
              builder: (ctx, constraints) {
                final widgetSize = Size(
                  constraints.maxWidth,
                  constraints.maxHeight,
                );
                return CustomPaint(
                  painter: BoundingBoxPainter(
                    results: _results,
                    imageSize: _imageSize,
                    widgetSize: widgetSize,
                    isFrontCamera: _isFrontCamera,
                  ),
                );
              },
            ),

          // ── Top bar ────────────────────────────────────────────────────
          SafeArea(child: _buildTopBar()),

          // ── Bottom stats ───────────────────────────────────────────────
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _buildBottomStats(),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    return Center(
      child: AspectRatio(
        aspectRatio: _cameraController!.value.aspectRatio,
        child: CameraPreview(_cameraController!),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          _GlassButton(
            icon: Icons.arrow_back_ios_new_rounded,
            onTap: () => Navigator.pop(context),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'LIVE DETECTION',
              style: TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w700,
                letterSpacing: 2.5,
              ),
            ),
          ),
          _StatusBadge(label: '$_fps FPS', color: const Color(0xFF00E5FF)),
          const SizedBox(width: 8),
          _StatusBadge(
            label: '${_results.length} obj',
            color: const Color(0xFF7C4DFF),
          ),
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
            begin: Alignment.bottomCenter,
            end: Alignment.topCenter,
            colors: [Colors.black.withOpacity(0.8), Colors.transparent],
          ),
        ),
        child: Center(
          child: Text(
            'Point camera at objects...',
            style: TextStyle(
              color: Colors.white.withOpacity(0.4),
              fontSize: 13,
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
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black.withOpacity(0.9), Colors.transparent],
        ),
      ),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: _results.take(5).map((r) {
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: const Color(0xFF00E5FF).withOpacity(0.4),
              ),
            ),
            child: Text(
              '${r.label} · ${(r.confidence * 100).toStringAsFixed(0)}% · ${r.distance}',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// ── Small reusable widgets ────────────────────────────────────────────────────

class _GlassButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const _GlassButton({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.15),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.white.withOpacity(0.2)),
        ),
        child: Icon(icon, color: Colors.white, size: 18),
      ),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final String label;
  final Color color;

  const _StatusBadge({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
