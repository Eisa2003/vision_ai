// FILE: lib/image_detection_screen.dart

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';

import 'services/detection_service.dart';
import 'services/distance_service.dart';
import 'widgets/bounding_box_painter.dart';

class ImageDetectionScreen extends StatefulWidget {
  const ImageDetectionScreen({super.key});

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  final _detectionService = DetectionService();
  final _distanceService = DistanceService();
  final _picker = ImagePicker();

  File? _selectedImage;
  Size _imageSize = Size.zero;
  List<DetectionResult> _results = [];
  bool _isProcessing = false;

  // The GlobalKey lets us measure the rendered image widget's size
  final _imageKey = GlobalKey();
  Size _renderedImageSize = Size.zero;

  @override
  void initState() {
    super.initState();
    _detectionService.initialize();
  }

  @override
  void dispose() {
    _detectionService.close();
    super.dispose();
  }

  Future<void> _pickAndDetect() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _selectedImage = File(picked.path);
      _results = [];
      _isProcessing = true;
    });

    // Decode image to get native size
    final decoded = await decodeImageFromList(await File(picked.path).readAsBytes());
    _imageSize = Size(decoded.width.toDouble(), decoded.height.toDouble());

    // Run detection
    final objects = await _detectionService.processFile(picked.path);

    final results = objects.map((obj) {
      return DetectionResult(
        boundingBox: obj.boundingBox,
        label: labelFor(obj),
        confidence: confidenceFor(obj),
        distance: _distanceService.estimate(obj.boundingBox, _imageSize),
      );
    }).toList();

    if (!mounted) return;
    setState(() {
      _results = results;
      _isProcessing = false;
    });

    // Measure the rendered image size after layout
    WidgetsBinding.instance.addPostFrameCallback((_) => _measureRenderedImage());
  }

  void _measureRenderedImage() {
    final ctx = _imageKey.currentContext;
    if (ctx == null) return;
    final box = ctx.findRenderObject() as RenderBox?;
    if (box == null) return;
    setState(() {
      _renderedImageSize = box.size;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF0A0E1A), Color(0xFF0D1B2A), Color(0xFF1A0A2E)],
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
          const Text(
            'IMAGE DETECTION',
            style: TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w700,
              letterSpacing: 2.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_selectedImage == null) {
      return _buildEmptyState();
    }
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
              border: Border.all(
                color: const Color(0xFF7C4DFF).withOpacity(0.4),
              ),
            ),
            child: const Icon(
              Icons.image_search_rounded,
              color: Color(0xFF7C4DFF),
              size: 48,
            ),
          ),
          const SizedBox(height: 24),
          const Text(
            'No image selected',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
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
            // ── Image + bounding box overlay ───────────────────────────
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                children: [
                  // The image
                  Image.file(
                    _selectedImage!,
                    key: _imageKey,
                    fit: BoxFit.contain,
                    width: double.infinity,
                  ),

                  // Processing shimmer overlay
                  if (_isProcessing)
                    Positioned.fill(
                      child: Container(
                        color: Colors.black.withOpacity(0.5),
                        child: const Center(
                          child: CircularProgressIndicator(
                            color: Color(0xFF7C4DFF),
                          ),
                        ),
                      ),
                    ),

                  // Bounding boxes
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

            // ── Results list ────────────────────────────────────────────
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
        ..._results.asMap().entries.map((entry) {
          final i = entry.key;
          final r = entry.value;
          return _ResultTile(result: r, index: i);
        }),
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
            style: TextStyle(
                color: Colors.white.withOpacity(0.5), fontSize: 13),
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
                _isProcessing ? Icons.hourglass_top_rounded : Icons.add_photo_alternate_rounded,
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
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              result.label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          _Chip(
            label: '${(result.confidence * 100).toStringAsFixed(0)}%',
            color: color,
          ),
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
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }
}
