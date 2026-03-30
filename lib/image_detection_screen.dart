// FILE: lib/image_detection_screen.dart

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

import 'providers.dart';
import 'services/yolo_detector.dart';
import 'widgets/bounding_box_painter.dart';
import 'services/base_detector.dart';

class ImageDetectionScreen extends ConsumerStatefulWidget {
  const ImageDetectionScreen({super.key});

  @override
  ConsumerState<ImageDetectionScreen> createState() =>
      _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends ConsumerState<ImageDetectionScreen> {
  final _picker = ImagePicker();

  File? _selectedImage;
  Size _imageSize = Size.zero;
  List<DetectionResult> _results = [];
  bool _isProcessing = false;
  final _imageKey = GlobalKey();
  Size _renderedImageSize = Size.zero;

  Future<void> _pickAndDetect() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _selectedImage = File(picked.path);
      _results = [];
      _isProcessing = true;
    });

    final decoded =
        await decodeImageFromList(await File(picked.path).readAsBytes());
    final imageSize = Size(decoded.width.toDouble(), decoded.height.toDouble());

    // Use whichever detector is currently active
    final detectorState = ref.read(activeDetectorProvider);

    if (detectorState is! AsyncData) {
      setState(() => _isProcessing = false);
      return;
    }

    final detector = detectorState.value!;
    final distanceSvc = ref.read(activeDistanceProvider);

    final detections = detector is YoloDetector
        ? await detector.detectFromFile(picked.path)
        : await detector.detect(InputImage.fromFilePath(picked.path));

    final results = detections.map((d) {
      return DetectionResult(
        boundingBox: d.boundingBox,
        label: d.label,
        confidence: d.confidence,
        distance: distanceSvc.estimate(d.boundingBox, imageSize),
      );
    }).toList();

    if (!mounted) return;
    setState(() {
      _imageSize = imageSize;
      _results = results;
      _isProcessing = false;
    });

    WidgetsBinding.instance
        .addPostFrameCallback((_) => _measureRenderedImage());
  }

  void _measureRenderedImage() {
    final ctx = _imageKey.currentContext;
    if (ctx == null) return;
    final box = ctx.findRenderObject() as RenderBox?;
    if (box == null) return;
    setState(() => _renderedImageSize = box.size);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0A0E1A),
              Color(0xFF0D1B2A),
              Color(0xFF1A0A2E),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildTopBar(),
              Expanded(child: _buildBody()),
              _buildBottomBar(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    final detectorName = ref.watch(activeDetectorTypeProvider).label;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.white.withOpacity(0.15)),
              ),
              child: const Icon(Icons.arrow_back_ios_new_rounded,
                  color: Colors.white, size: 18),
            ),
          ),
          const SizedBox(width: 14),
          const Expanded(
            child: Text(
              'IMAGE DETECTION',
              style: TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w700,
                letterSpacing: 2.5,
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0xFF7C4DFF).withOpacity(0.15),
              borderRadius: BorderRadius.circular(6),
              border:
                  Border.all(color: const Color(0xFF7C4DFF).withOpacity(0.5)),
            ),
            child: Text(
              detectorName,
              style: const TextStyle(
                color: Color(0xFF7C4DFF),
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_selectedImage == null) return _buildEmptyState();
    return _buildImageWithOverlay();
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: const Color(0xFF7C4DFF).withOpacity(0.1),
              borderRadius: BorderRadius.circular(24),
              border:
                  Border.all(color: const Color(0xFF7C4DFF).withOpacity(0.4)),
            ),
            child: const Icon(Icons.image_search_rounded,
                color: Color(0xFF7C4DFF), size: 48),
          ),
          const SizedBox(height: 24),
          const Text(
            'No image selected',
            style: TextStyle(
                color: Colors.white, fontSize: 18, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            'Tap the button below to pick\nan image from your gallery.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.45),
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageWithOverlay() {
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                children: [
                  Image.file(
                    _selectedImage!,
                    key: _imageKey,
                    fit: BoxFit.contain,
                    width: double.infinity,
                  ),
                  if (_isProcessing)
                    Positioned.fill(
                      child: Container(
                        color: Colors.black.withOpacity(0.5),
                        child: const Center(
                          child: CircularProgressIndicator(
                              color: Color(0xFF7C4DFF)),
                        ),
                      ),
                    ),
                  if (!_isProcessing &&
                      _results.isNotEmpty &&
                      _renderedImageSize != Size.zero)
                    Positioned.fill(
                      child: CustomPaint(
                        painter: BoundingBoxPainter(
                          results: _results,
                          imageSize: _imageSize,
                          widgetSize: _renderedImageSize,
                        ),
                      ),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            if (!_isProcessing && _results.isNotEmpty) _buildResultsList(),
            if (!_isProcessing && _results.isEmpty && _selectedImage != null)
              _buildNoDetections(),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsList() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Text(
            '${_results.length} OBJECT${_results.length != 1 ? 'S' : ''} DETECTED',
            style: const TextStyle(
              color: Color(0xFF7C4DFF),
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 2,
            ),
          ),
        ),
        ..._results
            .asMap()
            .entries
            .map((e) => _ResultTile(result: e.value, index: e.key)),
      ],
    );
  }

  Widget _buildNoDetections() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline_rounded,
              color: Colors.white.withOpacity(0.4), size: 18),
          const SizedBox(width: 10),
          Text(
            'No objects detected. Try a clearer image.',
            style:
                TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
      child: GestureDetector(
        onTap: _isProcessing ? null : _pickAndDetect,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          height: 56,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: _isProcessing
                  ? [Colors.grey.shade800, Colors.grey.shade700]
                  : [const Color(0xFF7C4DFF), const Color(0xFF5C35CC)],
            ),
            borderRadius: BorderRadius.circular(16),
            boxShadow: _isProcessing
                ? []
                : [
                    BoxShadow(
                      color: const Color(0xFF7C4DFF).withOpacity(0.4),
                      blurRadius: 20,
                      offset: const Offset(0, 6),
                    ),
                  ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                _isProcessing
                    ? Icons.hourglass_top_rounded
                    : Icons.add_photo_alternate_rounded,
                color: Colors.white,
                size: 22,
              ),
              const SizedBox(width: 10),
              Text(
                _isProcessing ? 'Detecting...' : 'Pick Image from Gallery',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultTile extends StatelessWidget {
  final DetectionResult result;
  final int index;
  const _ResultTile({required this.result, required this.index});

  @override
  Widget build(BuildContext context) {
    const colors = [Color(0xFF00E5FF), Color(0xFF7C4DFF)];
    final color = colors[index % colors.length];
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Row(
        children: [
          Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 12),
          Expanded(
            child: Text(result.label,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600)),
          ),
          _Chip(
              label: '${(result.confidence * 100).toStringAsFixed(0)}%',
              color: color),
          const SizedBox(width: 6),
          _Chip(label: result.distance, color: const Color(0xFF00BFA5)),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color color;
  const _Chip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Text(label,
          style: TextStyle(
              color: color, fontSize: 11, fontWeight: FontWeight.w700)),
    );
  }
}
