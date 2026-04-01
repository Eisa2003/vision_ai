// FILE: lib/services/yolo_detector.dart

import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/painting.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'base_detector.dart';

class YoloDetector extends BaseDetector {
  static const _modelPath = 'assets/models/yolov8n_float32.tflite';
  static const _labelPath = 'assets/models/coco_labels.txt';

  static const int _inputSize = 320;
  static const double _confidenceThreshold = 0.45;
  static const double _iouThreshold = 0.45;
  static const int _maxResults = 10;

  bool _initialized = false;
  List<String> _labels = [];
  Interpreter? _interpreter;

  @override
  String get name => 'YOLOv8n';

  @override
  String get description => 'TFLite · 80 COCO classes · standard interpreter';

  // ─── Lifecycle ────────────────────────────────────────────────────────────

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    // Load COCO labels
    try {
      final raw = await rootBundle.loadString(_labelPath);
      _labels = raw
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();
      print('✅ Labels loaded: ${_labels.length} classes');
      print('   First 5: ${_labels.take(5).toList()}');
    } catch (e) {
      print('⚠️  Labels load failed: $e — using class_N fallback');
      _labels = List.generate(80, (i) => 'class_$i');
    }

    // Load TFLite model
    print('🔄 Loading TFLite model from $_modelPath ...');
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
    _interpreter!.allocateTensors();

    // ── Print tensor shapes so we know the model's actual layout ──────────
    final inputTensor  = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);
    print('✅ Model loaded');
    print('   Input  tensor shape : ${inputTensor.shape}   type: ${inputTensor.type}');
    print('   Output tensor shape : ${outputTensor.shape}  type: ${outputTensor.type}');
    // Expected:
    //   Input  [1, 640, 640, 3]
    //   Output [1, 84, 8400]   (or [1, 8400, 84] for some exports)

    _initialized = true;
  }

  @override
  Future<void> close() async {
    if (!_initialized) return;
    _interpreter?.close();
    _interpreter = null;
    _initialized = false;
  }

  // ─── Public detect entry-points ───────────────────────────────────────────

  @override
  Future<List<Detection>> detect(dynamic image) async => [];

  Future<List<Detection>> detectFromBytes({
    required Uint8List bytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) async {
    if (!_initialized || _interpreter == null) {
      print('❌ detectFromBytes: not initialized');
      return [];
    }

    print('📷 detectFromBytes — frame ${imageW}x${imageH}, rotation=$rotation, bytes=${bytes.length}');

    try {
      print('🔄 Converting NV21 → RGB ...');
      final rgbImage = _nv21ToRgb(bytes, imageW, imageH, rotation);
      print('✅ RGB image size: ${rgbImage.width}x${rgbImage.height}');

      print('🔄 Converting RGB → Float32 ...');
      final input = _imageToFloat32(rgbImage);
      print('✅ Float32 buffer length: ${input.length}  (expected ${_inputSize * _inputSize * 3})');

      return _runInference(input, imageW.toDouble(), imageH.toDouble());
    } catch (e, st) {
      print('❌ detectFromBytes error: $e');
      print(st);
      return [];
    }
  }

  Future<List<Detection>> detectFromFile(String filePath) async {
    if (!_initialized || _interpreter == null) {
      print('❌ detectFromFile: not initialized');
      return [];
    }

    print('🖼️  detectFromFile: $filePath');

    try {
      final fileBytes = await rootBundle.load(filePath);
      final decoded = img.decodeImage(fileBytes.buffer.asUint8List());
      if (decoded == null) {
        print('❌ Could not decode image at $filePath');
        return [];
      }

      print('✅ Decoded image: ${decoded.width}x${decoded.height}');

      final w = decoded.width.toDouble();
      final h = decoded.height.toDouble();
      final resized = img.copyResize(decoded,
          width: _inputSize,
          height: _inputSize,
          interpolation: img.Interpolation.linear);
      final input = _imageToFloat32(resized);

      print('✅ Float32 buffer length: ${input.length}');
      return _runInference(input, w, h);
    } catch (e, st) {
      print('❌ detectFromFile error: $e');
      print(st);
      return [];
    }
  }

  // ─── Inference ────────────────────────────────────────────────────────────

  List<Detection> _runInference(
      Float32List input, double origW, double origH) {
    final interpreter = _interpreter!;

    // ── Step 1: confirm tensor shapes ─────────────────────────────────────
    final inputShape  = interpreter.getInputTensor(0).shape;
    final outputShape = interpreter.getOutputTensor(0).shape;
    print('🔢 Input  shape: $inputShape');
    print('🔢 Output shape: $outputShape');

    // ── Step 2: build nested input list [1][640][640][3] ──────────────────
    // tflite_flutter requires a nested List, not a flat Float32List, when
    // you use interpreter.run().  Build it directly from the flat buffer.
    print('🔄 Building nested input list ...');
    final int h = inputShape[1]; // 640
    final int w = inputShape[2]; // 640
    final int c = inputShape[3]; // 3

    // Verify our float buffer is the right size
    if (input.length != h * w * c) {
      print('❌ Buffer size mismatch: got ${input.length}, expected ${h * w * c}');
      return [];
    }

    // Build [1][H][W][C] nested list
    final inputList = List.generate(
      1,
      (_) => List.generate(
        h,
        (row) => List.generate(
          w,
          (col) {
            final base = (row * w + col) * c;
            return [input[base], input[base + 1], input[base + 2]];
          },
        ),
      ),
    );
    print('✅ Nested input list built: 1 × $h × $w × $c');

    // ── Step 3: build output buffer ────────────────────────────────────────
    // outputShape is typically [1, 84, 8400]
    // We need a mutable nested list of the same shape filled with zeros.
    final int dim1 = outputShape[1]; // 84
    final int dim2 = outputShape[2]; // 8400
    final outputList = List.generate(
      1,
      (_) => List.generate(
        dim1,
        (_) => List.filled(dim2, 0.0),
      ),
    );
    print('✅ Output buffer built: 1 × $dim1 × $dim2');

    // ── Step 4: run inference ──────────────────────────────────────────────
    print('🚀 Running interpreter.run() ...');
    try {
      interpreter.run(inputList, outputList);
      print('✅ interpreter.run() completed');
    } catch (e, st) {
      print('❌ interpreter.run() threw: $e');
      print(st);
      return [];
    }

    // ── Step 5: sanity-check output values ────────────────────────────────
    // Sample a few values from the output to confirm it's not all zeros
    final sample = <double>[];
    for (int i = 0; i < min(5, dim2); i++) {
      sample.add((outputList[0][0] as List)[i] as double); // first row = cx values
    }
    print('🔍 Output sample (first 5 cx values): $sample');

    // Check if all zeros (model didn't write output)
    final flat = outputList[0].expand((row) => row as List<double>);
    final nonZero = flat.where((v) => v != 0.0).length;
    print('🔍 Non-zero output values: $nonZero / ${dim1 * dim2}');
    if (nonZero == 0) {
      print('⚠️  All output values are zero — model may not have run correctly');
    }

    // ── Step 6: parse detections ───────────────────────────────────────────
    print('🔄 Parsing output (origW=$origW, origH=$origH) ...');
    final rawOutput = outputList[0] as List;
    final detections = _parseOutput(rawOutput, origW, origH);

    print('✅ _runInference done — ${detections.length} detection(s) returned');
    return detections;
  }

  // ─── Output parsing ───────────────────────────────────────────────────────

  List<Detection> _parseOutput(
      List<dynamic> output, double origW, double origH) {
    // output layout: [84][8400]
    // rows 0-3  → cx, cy, w, h  (normalised 0–1 relative to _inputSize)
    // rows 4-83 → class scores

    final int numAnchors = (output[0] as List).length;
    final int numRows    = output.length;
    print('🔍 Parsing: numRows=$numRows, numAnchors=$numAnchors');

    // Quick confidence scan — find the max score in the whole output
    // to confirm the model is producing non-trivial values
    double globalMax = 0;
    for (int c = 4; c < numRows; c++) {
      for (int a = 0; a < numAnchors; a++) {
        final v = (output[c] as List)[a] as double;
        if (v > globalMax) globalMax = v;
      }
    }
    print('🔍 Global max class score across all anchors: ${globalMax.toStringAsFixed(4)}');
    print('   Confidence threshold: $_confidenceThreshold');
    if (globalMax < _confidenceThreshold) {
      print('⚠️  No anchor exceeds threshold — try lowering _confidenceThreshold');
    }

    final List<_RawBox> candidates = [];

    for (int a = 0; a < numAnchors; a++) {
      double maxScore = 0;
      int bestClass = 0;
      for (int c = 4; c < numRows; c++) {
        final score = (output[c] as List)[a] as double;
        if (score > maxScore) {
          maxScore = score;
          bestClass = c - 4;
        }
      }
      if (maxScore < _confidenceThreshold) continue;

      final cx = (output[0] as List)[a] as double;
      final cy = (output[1] as List)[a] as double;
      final bw = (output[2] as List)[a] as double;
      final bh = (output[3] as List)[a] as double;

      // YOLOv8 TFLite export: coords are in pixel space relative to _inputSize
      // Normalise to [0,1] then scale to original image dimensions
      final left   = (cx - bw / 2) / _inputSize * origW;
      final top    = (cy - bh / 2) / _inputSize * origH;
      final width  = bw / _inputSize * origW;
      final height = bh / _inputSize * origH;

      candidates.add(_RawBox(
        rect: Rect.fromLTWH(left, top, width, height),
        confidence: maxScore,
        classIndex: bestClass,
      ));
    }

    print('🔍 Candidates above threshold: ${candidates.length}');

    final kept = _nms(candidates);
    print('🔍 After NMS: ${kept.length}');

    return kept
        .take(_maxResults)
        .map((b) => Detection(
              boundingBox: b.rect,
              label: _labelForIndex(b.classIndex),
              confidence: b.confidence,
            ))
        .toList();
  }

  // ─── NMS ──────────────────────────────────────────────────────────────────

  List<_RawBox> _nms(List<_RawBox> boxes) {
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <_RawBox>[];
    final suppressed = List.filled(boxes.length, false);
    for (int i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      kept.add(boxes[i]);
      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(boxes[i].rect, boxes[j].rect) > _iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final inter = a.intersect(b);
    if (inter.isEmpty) return 0;
    final interArea = inter.width * inter.height;
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea <= 0 ? 0 : interArea / unionArea;
  }

  // ─── Image helpers ────────────────────────────────────────────────────────

  img.Image _nv21ToRgb(
      Uint8List nv21, int width, int height, int rotation) {
    // Manual NV21 → RGB conversion
    // NV21 layout: Y plane (width×height bytes) followed by interleaved VU plane
    final ySize  = width * height;
    final rgbImg = img.Image(width: width, height: height);

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        final yIndex  = row * width + col;
        // VU plane starts at ySize; each 2×2 Y block shares one VU pair
        final uvIndex = ySize + (row ~/ 2) * width + (col & ~1);

        final Y = nv21[yIndex]  & 0xFF;
        final V = nv21[uvIndex] & 0xFF;      // NV21: V comes before U
        final U = nv21[uvIndex + 1] & 0xFF;

        // BT.601 full-range conversion
        final r = (Y + 1.402 * (V - 128)).round().clamp(0, 255);
        final g = (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)).round().clamp(0, 255);
        final b = (Y + 1.772 * (U - 128)).round().clamp(0, 255);

        rgbImg.setPixelRgb(col, row, r, g, b);
      }
    }

    // Rotate
    img.Image rotated = rgbImg;
    if (rotation == 90)       rotated = img.copyRotate(rgbImg, angle: 90);
    else if (rotation == 180) rotated = img.copyRotate(rgbImg, angle: 180);
    else if (rotation == 270) rotated = img.copyRotate(rgbImg, angle: 270);

    return img.copyResize(rotated,
        width: _inputSize,
        height: _inputSize,
        interpolation: img.Interpolation.linear);
  }

  Float32List _imageToFloat32(img.Image image) {
    final buffer = Float32List(image.width * image.height * 3);
    int idx = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        buffer[idx++] = pixel.r / 255.0;
        buffer[idx++] = pixel.g / 255.0;
        buffer[idx++] = pixel.b / 255.0;
      }
    }
    return buffer;
  }

  String _labelForIndex(int index) {
    if (index >= 0 && index < _labels.length) return _labels[index];
    return 'object';
  }
}

class _RawBox {
  final Rect rect;
  final double confidence;
  final int classIndex;
  const _RawBox({
    required this.rect,
    required this.confidence,
    required this.classIndex,
  });
}
