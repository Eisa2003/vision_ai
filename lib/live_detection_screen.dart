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

  BaseDetector? _detector;

  // ─── Init ─────────────────────────────────────────────────────────────────

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
      imageFormatGroup: ImageFormatGroup.nv21,
    );

    await _cameraController!.initialize();
    if (!mounted) return;

    setState(() => _isCameraReady = true);
    _cameraController!.startImageStream(_onCameraFrame);
  }

  // ─── Camera frame callback ────────────────────────────────────────────────

  Future<void> _onCameraFrame(CameraImage image) async {
    // Skip frame if a previous inference is still in flight
    if (_isProcessing) return;

    // Resolve the active detector from Riverpod
    final detectorState = ref.read(activeDetectorProvider);
    detectorState.when(
      data: (d) => _detector = d,
      loading: () => null,
      error: (e, st) => print('❌ Detector error: $e'),
    );

    if (_detector == null) return;

    final detector = _detector!;

    // ── Snapshot bytes immediately on the main isolate ─────────────────────
    // The camera reuses its plane buffers — without copying here, the
    // background isolate would read garbage by the time it runs.
    final allBytes = BytesBuilder();
    for (final plane in image.planes) {
      allBytes.add(Uint8List.fromList(plane.bytes));
    }
    final imageW = image.width;
    final imageH = image.height;
    final rotation = _cameraController!.description.sensorOrientation;

    // Account for sensor rotation when reporting frame size to the painter
    final frameSize = (rotation == 90 || rotation == 270)
        ? Size(imageH.toDouble(), imageW.toDouble())
        : Size(imageW.toDouble(), imageH.toDouble());

    // ── Release the lock BEFORE awaiting inference ─────────────────────────
    // This keeps the camera stream running while YOLO runs in the background.
    // ML Kit manages its own threading so this helps it too.
    _isProcessing = true;

    try {
      final distanceSvc = ref.read(activeDistanceProvider);

      List<Detection> detections;

      if (detector is YoloDetector) {
        // Runs in a background isolate — camera stays smooth
        detections = await detector.detectFromBytesIsolated(
          bytes: allBytes.toBytes(),
          imageW: imageW,
          imageH: imageH,
          rotation: rotation,
        );
      } else {
        // ML Kit path — uses InputImage wrapper
        final inputImage = _buildInputImage(image);
        if (inputImage == null) return;
        detections = await detector.detect(inputImage);
      }

      if (!mounted) return;

      final results = detections.map((d) {
        return DetectionResult(
          boundingBox: d.boundingBox,
          label: d.label,
          confidence: d.confidence,
          distance: distanceSvc.estimate(d.boundingBox, frameSize),
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
        _imageSize = frameSize;
      });
    } catch (e) {
      print('❌ Frame processing error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  // ─── ML Kit InputImage builder ────────────────────────────────────────────

  InputImage? _buildInputImage(CameraImage image) {
    final camera = _cameraController!.description;
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

  // ─── Dispose ──────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    super.dispose();
  }

  // ─── UI ───────────────────────────────────────────────────────────────────

   @override
  Widget build(BuildContext context) {
    final detectorType = ref.watch(activeDetectorTypeProvider);
    final screenH = MediaQuery.of(context).size.height;
 
    return Scaffold(
      backgroundColor: const Color(0xFF080C14),
      body: SafeArea(
        child: Column(
          children: [
            // ── Top bar ──────────────────────────────────────────────────────
            _buildTopBar(detectorType),
 
            // ── Camera viewport (≈70 % of screen) ────────────────────────────
            Expanded(
              flex: 70,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: _buildCameraViewport(),
              ),
            ),
 
            // ── Bottom panel ──────────────────────────────────────────────────
            Expanded(
              flex: 30,
              child: _buildBottomPanel(detectorType),
            ),
          ],
        ),
      ),
    );
  }
 
  // ── Camera viewport with rounded corners + overlay ────────────────────────
 
  Widget _buildCameraViewport() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Camera feed
          if (_isCameraReady && _cameraController != null)
            CameraPreview(_cameraController!)
          else
            Container(
              color: const Color(0xFF0D1520),
              child: const Center(
                child: CircularProgressIndicator(
                  color: Color(0xFF00E5FF),
                  strokeWidth: 2,
                ),
              ),
            ),
 
          // Subtle vignette to frame the feed
          IgnorePointer(
            child: Container(
              decoration: BoxDecoration(
                gradient: RadialGradient(
                  center: Alignment.center,
                  radius: 1.1,
                  colors: [
                    Colors.transparent,
                    Colors.black.withOpacity(0.35),
                  ],
                ),
              ),
            ),
          ),
 
          // Bounding boxes
          if (_isCameraReady && _imageSize != Size.zero)
            LayoutBuilder(builder: (ctx, constraints) {
              return CustomPaint(
                painter: BoundingBoxPainter(
                  results: _results,
                  imageSize: _imageSize,
                  widgetSize: Size(constraints.maxWidth, constraints.maxHeight),
                  isFrontCamera: _isFrontCamera,
                ),
              );
            }),
 
          // Corner-bracket overlay (purely decorative, signals "scanning")
          const IgnorePointer(child: _ScannerBrackets()),
 
          // Live pill — top-left of viewport
          Positioned(
            top: 12,
            left: 12,
            child: _LivePill(isActive: _isCameraReady && _results.isNotEmpty),
          ),
 
          // Object count — top-right of viewport
          Positioned(
            top: 12,
            right: 12,
            child: _CountBadge(count: _results.length),
          ),
        ],
      ),
    );
  }
 
  // ── Top bar ────────────────────────────────────────────────────────────────
 
  Widget _buildTopBar(DetectorType detectorType) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
      child: Row(
        children: [
          // Back button
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.07),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white.withOpacity(0.1)),
              ),
              child: const Icon(
                Icons.arrow_back_ios_new_rounded,
                color: Colors.white,
                size: 16,
              ),
            ),
          ),
 
          const SizedBox(width: 14),
 
          // Title
          const Expanded(
            child: Text(
              'LIVE DETECTION',
              style: TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w800,
                letterSpacing: 3,
              ),
            ),
          ),
 
          // FPS badge
          _TopBadge(
            label: '$_fps',
            sublabel: 'FPS',
            color: _fpsColor,
          ),
        ],
      ),
    );
  }
 
  Color get _fpsColor {
    if (_fps >= 20) return const Color(0xFF00E5A0);
    if (_fps >= 10) return const Color(0xFFFFB300);
    return const Color(0xFFFF4B6E);
  }
 
  // ── Bottom panel ───────────────────────────────────────────────────────────
 
  Widget _buildBottomPanel(DetectorType detectorType) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1520),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.07)),
      ),
      child: Column(
        children: [
          // ── Model row ───────────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
            child: Row(
              children: [
                // Pulsing dot
                _PulsingDot(active: _isCameraReady),
                const SizedBox(width: 8),
                Text(
                  detectorType.label.toUpperCase(),
                  style: const TextStyle(
                    color: Color(0xFF00E5FF),
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 2,
                  ),
                ),
                const Spacer(),
                Text(
                  'ACTIVE MODEL',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.25),
                    fontSize: 9,
                    letterSpacing: 1.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
 
          const SizedBox(height: 10),
 
          // ── Divider ─────────────────────────────────────────────────────
          Divider(
            height: 1,
            color: Colors.white.withOpacity(0.06),
            indent: 18,
            endIndent: 18,
          ),
 
          const SizedBox(height: 10),
 
          // ── Detection chips ─────────────────────────────────────────────
          Expanded(
            child: _results.isEmpty
                ? Center(
                    child: Text(
                      'Point camera at objects',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.2),
                        fontSize: 12,
                        letterSpacing: 0.5,
                      ),
                    ),
                  )
                : Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    child: Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _results.take(6).map((r) {
                        return _DetectionChip(result: r);
                      }).toList(),
                    ),
                  ),
          ),
 
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
 
// ─────────────────────────────────────────────────────────────────────────────
// Supporting widgets
// ─────────────────────────────────────────────────────────────────────────────
 
/// Animated "LIVE" pill shown inside the camera viewport.
class _LivePill extends StatefulWidget {
  final bool isActive;
  const _LivePill({required this.isActive});
 
  @override
  State<_LivePill> createState() => _LivePillState();
}
 
class _LivePillState extends State<_LivePill>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _fade;
 
  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _fade = Tween<double>(begin: 0.4, end: 1.0).animate(_ctrl);
  }
 
  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }
 
  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fade,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.55),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: widget.isActive
                ? const Color(0xFF00E5A0).withOpacity(0.7)
                : Colors.white24,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 6,
              height: 6,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: widget.isActive
                    ? const Color(0xFF00E5A0)
                    : Colors.white38,
              ),
            ),
            const SizedBox(width: 5),
            Text(
              'LIVE',
              style: TextStyle(
                color: widget.isActive
                    ? const Color(0xFF00E5A0)
                    : Colors.white54,
                fontSize: 9,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
 
/// Object count badge inside the camera viewport.
class _CountBadge extends StatelessWidget {
  final int count;
  const _CountBadge({required this.count});
 
  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: count > 0
              ? const Color(0xFF00E5FF).withOpacity(0.6)
              : Colors.white12,
        ),
      ),
      child: Text(
        '$count OBJ',
        style: TextStyle(
          color: count > 0 ? const Color(0xFF00E5FF) : Colors.white38,
          fontSize: 9,
          fontWeight: FontWeight.w800,
          letterSpacing: 1.5,
        ),
      ),
    );
  }
}
 
/// FPS / model badge for the top bar.
class _TopBadge extends StatelessWidget {
  final String label;
  final String sublabel;
  final Color color;
  const _TopBadge({
    required this.label,
    required this.sublabel,
    required this.color,
  });
 
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 15,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(width: 3),
          Text(
            sublabel,
            style: TextStyle(
              color: color.withOpacity(0.6),
              fontSize: 9,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }
}
 
/// Pulsing dot indicator next to the model name.
class _PulsingDot extends StatefulWidget {
  final bool active;
  const _PulsingDot({required this.active});
 
  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}
 
class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _scale;
 
  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _scale = Tween<double>(begin: 0.7, end: 1.0).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
  }
 
  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }
 
  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _scale,
      child: Container(
        width: 7,
        height: 7,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: widget.active
              ? const Color(0xFF00E5FF)
              : Colors.white24,
          boxShadow: widget.active
              ? [
                  BoxShadow(
                    color: const Color(0xFF00E5FF).withOpacity(0.6),
                    blurRadius: 6,
                    spreadRadius: 1,
                  )
                ]
              : null,
        ),
      ),
    );
  }
}
 
/// Single detection result chip shown in the bottom panel.
class _DetectionChip extends StatelessWidget {
  final DetectionResult result;
  const _DetectionChip({required this.result});
 
  // Deterministic colour per label
  Color _labelColor() {
    const palette = [
      Color(0xFF00E5FF),
      Color(0xFF7C4DFF),
      Color(0xFF00E5A0),
      Color(0xFFFF6B6B),
      Color(0xFFFFB300),
      Color(0xFFFF4081),
    ];
    return palette[result.label.hashCode.abs() % palette.length];
  }
 
  @override
  Widget build(BuildContext context) {
    final color = _labelColor();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            result.label,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '${(result.confidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: Colors.white.withOpacity(0.45),
              fontSize: 10,
              fontWeight: FontWeight.w500,
            ),
          ),
          if (result.distance.isNotEmpty) ...[
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 5),
              child: Text(
                '·',
                style: TextStyle(color: Colors.white.withOpacity(0.2)),
              ),
            ),
            Text(
              result.distance,
              style: const TextStyle(
                color: Color(0xFF00E5FF),
                fontSize: 10,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
 
/// Four corner brackets drawn over the camera feed — pure decoration.
class _ScannerBrackets extends StatelessWidget {
  const _ScannerBrackets();
 
  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _BracketPainter());
  }
}
 
class _BracketPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF00E5FF).withOpacity(0.55)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
 
    const len = 22.0;
    const pad = 16.0;
 
    // Top-left
    canvas.drawLine(Offset(pad, pad + len), Offset(pad, pad), paint);
    canvas.drawLine(Offset(pad, pad), Offset(pad + len, pad), paint);
    // Top-right
    canvas.drawLine(Offset(size.width - pad - len, pad),
        Offset(size.width - pad, pad), paint);
    canvas.drawLine(Offset(size.width - pad, pad),
        Offset(size.width - pad, pad + len), paint);
    // Bottom-left
    canvas.drawLine(Offset(pad, size.height - pad - len),
        Offset(pad, size.height - pad), paint);
    canvas.drawLine(Offset(pad, size.height - pad),
        Offset(pad + len, size.height - pad), paint);
    // Bottom-right
    canvas.drawLine(Offset(size.width - pad - len, size.height - pad),
        Offset(size.width - pad, size.height - pad), paint);
    canvas.drawLine(Offset(size.width - pad, size.height - pad - len),
        Offset(size.width - pad, size.height - pad), paint);
  }
 
  @override
  bool shouldRepaint(_BracketPainter _) => false;
}
 
// ─── Shared small widgets (unchanged) ────────────────────────────────────────
 
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
