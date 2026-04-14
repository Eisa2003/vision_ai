// FILE: lib/services/yolo_detector.dart

import 'dart:async';
import 'dart:isolate';
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

  static const int    _inputSize            = 320;
  static const double _confidenceThreshold  = 0.45;
  static const double _iouThreshold         = 0.45;
  static const int    _maxResults           = 10;

  bool         _initialized = false;
  List<String> _labels      = [];
  Interpreter? _interpreter;
  Uint8List?   _modelBytes;

  // ── Persistent background isolate ────────────────────────────────────────
  Isolate?   _isolate;
  SendPort?  _sendPort;          // we send jobs to the isolate through this
  ReceivePort? _receivePort;     // isolate sends results back through this
  // Completer waiting for the current inference result
  _InferencePending? _pending;

  @override String get name        => 'YOLOv8n';
  @override String get description => 'TFLite · 80 COCO classes · isolate inference';

  // ─── Lifecycle ────────────────────────────────────────────────────────────

  @override
  Future<void> initialize() async {
    if (_initialized) return;

    // 1. Labels
    try {
      final raw = await rootBundle.loadString(_labelPath);
      _labels = raw.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
      print('✅ Labels loaded: ${_labels.length} classes');
    } catch (e) {
      print('⚠️  Labels fallback: $e');
      _labels = List.generate(80, (i) => 'class_$i');
    }

    // 2. Raw model bytes (needed by the isolate)
    print('🔄 Loading model bytes...');
    final data = await rootBundle.load(_modelPath);
    _modelBytes = data.buffer.asUint8List();

    // 3. Main-isolate interpreter (used by detectFromBytes directly if needed)
    final opts = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(_modelPath, options: opts);
    _interpreter!.allocateTensors();
    print('✅ Main interpreter ready  '
        'in=${_interpreter!.getInputTensor(0).shape}  '
        'out=${_interpreter!.getOutputTensor(0).shape}');

    // 4. Spin up the persistent background isolate
    await _spawnIsolate();

    _initialized = true;
  }

  /// Spawns the long-lived inference isolate and waits until it signals ready.
  Future<void> _spawnIsolate() async {
    _receivePort = ReceivePort();

    // The isolate will send us its own SendPort first, then inference results.
    final ready = _receivePort!.asBroadcastStream();

    _isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateStartup(
        mainSendPort: _receivePort!.sendPort,
        modelBytes:   _modelBytes!,
        labels:       _labels,
      ),
    );

    // First message from the isolate is its SendPort
    _sendPort = await ready.first as SendPort;
    print('✅ Background inference isolate ready');

    // Listen for subsequent inference results
    ready.listen((message) {
      if (message is List<Map<String, dynamic>>) {
        _pending?.complete(message);
        _pending = null;
      } else if (message is String) {
        // Error string from isolate
        _pending?.completeError(message);
        _pending = null;
      }
    });
  }

  @override
  Future<void> close() async {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate     = null;
    _receivePort?.close();
    _receivePort = null;
    _sendPort    = null;
    _pending     = null;
    _interpreter?.close();
    _interpreter = null;
    _initialized = false;
  }

  // ─── Public API ───────────────────────────────────────────────────────────

  @override
  Future<List<Detection>> detect(dynamic image) async => [];

  /// Runs inference on the persistent background isolate.
  Future<List<Detection>> detectFromBytesIsolated({
    required Uint8List bytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) async {
    if (_sendPort == null) {
      print('❌ Isolate not ready');
      return [];
    }
    if (_pending != null) {
      // Previous frame still in flight — skip this one
      return [];
    }

    _pending = _InferencePending();
    _sendPort!.send(_InferenceJob(
      imageBytes: bytes,
      imageW:     imageW,
      imageH:     imageH,
      rotation:   rotation,
    ));

    try {
      final maps = await _pending!.future;
      return maps.map((m) => Detection(
        boundingBox: Rect.fromLTRB(
          m['left']   as double,
          m['top']    as double,
          m['right']  as double,
          m['bottom'] as double,
        ),
        label:      m['label']      as String,
        confidence: m['confidence'] as double,
      )).toList();
    } catch (e) {
      print('❌ Isolate inference error: $e');
      return [];
    }
  }

  /// Direct main-isolate detection (used internally + by detectFromFile).
  Future<List<Detection>> detectFromBytes({
    required Uint8List bytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) async {
    if (!_initialized || _interpreter == null) return [];
    try {
      final rgb   = _nv21ToRgb(bytes, imageW, imageH, rotation);
      final input = _imageToFloat32(rgb);
      return _runInference(_interpreter!, input, imageW.toDouble(), imageH.toDouble());
    } catch (e, st) {
      print('❌ detectFromBytes: $e\n$st');
      return [];
    }
  }

  Future<List<Detection>> detectFromFile(String filePath) async {
    if (!_initialized || _interpreter == null) return [];
    try {
      final bytes   = await rootBundle.load(filePath);
      final decoded = img.decodeImage(bytes.buffer.asUint8List());
      if (decoded == null) return [];
      final resized = img.copyResize(decoded,
          width: _inputSize, height: _inputSize,
          interpolation: img.Interpolation.linear);
      final input = _imageToFloat32(resized);
      return _runInference(_interpreter!, input,
          decoded.width.toDouble(), decoded.height.toDouble());
    } catch (e, st) {
      print('❌ detectFromFile: $e\n$st');
      return [];
    }
  }

  // ─── Inference (shared by both main and background isolate) ──────────────

  static List<Detection> _runInference(
      Interpreter interpreter, Float32List input,
      double origW, double origH) {
    final inShape  = interpreter.getInputTensor(0).shape;
    final outShape = interpreter.getOutputTensor(0).shape;

    final h = inShape[1], w = inShape[2], c = inShape[3];
    if (input.length != h * w * c) return [];

    final inputList = List.generate(1, (_) =>
      List.generate(h, (row) =>
        List.generate(w, (col) {
          final base = (row * w + col) * c;
          return [input[base], input[base + 1], input[base + 2]];
        })));

    final dim1 = outShape[1], dim2 = outShape[2];
    final outputList = List.generate(1, (_) =>
        List.generate(dim1, (_) => List.filled(dim2, 0.0)));

    try {
      interpreter.run(inputList, outputList);
    } catch (e) {
      print('❌ interpreter.run(): $e');
      return [];
    }

    return _parseOutput(outputList[0] as List, origW, origH);
  }

  static List<Detection> _parseOutput(
      List<dynamic> output, double origW, double origH) {
    final numAnchors = (output[0] as List).length;
    final numRows    = output.length;

    double globalMax = 0;
    for (int c = 4; c < numRows; c++) {
      for (int a = 0; a < numAnchors; a++) {
        final v = (output[c] as List)[a] as double;
        if (v > globalMax) globalMax = v;
      }
    }
    if (globalMax < _confidenceThreshold) {
      print('⚠️  Max score ${globalMax.toStringAsFixed(3)} < threshold $_confidenceThreshold');
    }

    final candidates = <_RawBox>[];
    for (int a = 0; a < numAnchors; a++) {
      double maxScore  = 0;
      int    bestClass = 0;
      for (int c = 4; c < numRows; c++) {
        final score = (output[c] as List)[a] as double;
        if (score > maxScore) { maxScore = score; bestClass = c - 4; }
      }
      if (maxScore < _confidenceThreshold) continue;

      final cx = (output[0] as List)[a] as double;
      final cy = (output[1] as List)[a] as double;
      final bw = (output[2] as List)[a] as double;
      final bh = (output[3] as List)[a] as double;

      candidates.add(_RawBox(
        rect: Rect.fromLTWH(
          (cx - bw / 2),
          (cy - bh / 2),
          bw,
          bh,
        ),
        confidence: maxScore,
        classIndex: bestClass,
      ));
    }

    final kept = _nms(candidates);
    print('🔍 Candidates: ${candidates.length} → NMS kept: ${kept.length}');

    return kept.take(_maxResults).map((b) => Detection(
      boundingBox: Rect.fromLTWH(
        b.rect.left   * origW,
        b.rect.top    * origH,
        b.rect.width  * origW,
        b.rect.height * origH,
      ),
      label:      _labelList.isNotEmpty && b.classIndex < _labelList.length
                    ? _labelList[b.classIndex]
                    : 'object',
      confidence: b.confidence,
    )).toList();
  }

  // Static label list — set by isolate entry so _parseOutput can access it
  static List<String> _labelList = [];

  static List<_RawBox> _nms(List<_RawBox> boxes) {
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept       = <_RawBox>[];
    final suppressed = List.filled(boxes.length, false);
    for (int i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;
      kept.add(boxes[i]);
      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed[j]) continue;
        if (_iouStatic(boxes[i].rect, boxes[j].rect) > _iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  static double _iouStatic(Rect a, Rect b) {
    final inter = a.intersect(b);
    if (inter.isEmpty) return 0;
    final interArea = inter.width * inter.height;
    final unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea <= 0 ? 0 : interArea / unionArea;
  }

  // ─── Image helpers ────────────────────────────────────────────────────────

  static img.Image _nv21ToRgb(
      Uint8List nv21, int width, int height, int rotation) {
    final ySize  = width * height;
    final rgbImg = img.Image(width: width, height: height);
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        final yIdx  = row * width + col;
        final uvIdx = ySize + (row ~/ 2) * width + (col & ~1);
        final Y = nv21[yIdx]      & 0xFF;
        final V = nv21[uvIdx]     & 0xFF;
        final U = nv21[uvIdx + 1] & 0xFF;
        final r = (Y + 1.402    * (V - 128))                        .round().clamp(0, 255);
        final g = (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)).round().clamp(0, 255);
        final b = (Y + 1.772    * (U - 128))                        .round().clamp(0, 255);
        rgbImg.setPixelRgb(col, row, r, g, b);
      }
    }
    img.Image rotated = rgbImg;
    if      (rotation == 90)  rotated = img.copyRotate(rgbImg, angle: 90);
    else if (rotation == 180) rotated = img.copyRotate(rgbImg, angle: 180);
    else if (rotation == 270) rotated = img.copyRotate(rgbImg, angle: 270);
    return img.copyResize(rotated,
        width: _inputSize, height: _inputSize,
        interpolation: img.Interpolation.linear);
  }

  static Float32List _imageToFloat32(img.Image image) {
    final buffer = Float32List(image.width * image.height * 3);
    int idx = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final p = image.getPixel(x, y);
        buffer[idx++] = p.r / 255.0;
        buffer[idx++] = p.g / 255.0;
        buffer[idx++] = p.b / 255.0;
      }
    }
    return buffer;
  }
}

// ─── Internal types ───────────────────────────────────────────────────────────

class _RawBox {
  final Rect   rect;
  final double confidence;
  final int    classIndex;
  const _RawBox({required this.rect, required this.confidence, required this.classIndex});
}

class _InferencePending {
  final _completer = Completer<List<Map<String, dynamic>>>();
  Future<List<Map<String, dynamic>>> get future => _completer.future;
  void complete(List<Map<String, dynamic>> v) => _completer.complete(v);
  void completeError(Object e) => _completer.completeError(e);
}

// ─── Isolate messages ─────────────────────────────────────────────────────────

class _IsolateStartup {
  final SendPort  mainSendPort;
  final Uint8List modelBytes;
  final List<String> labels;
  const _IsolateStartup({
    required this.mainSendPort,
    required this.modelBytes,
    required this.labels,
  });
}

class _InferenceJob {
  final Uint8List imageBytes;
  final int imageW, imageH, rotation;
  const _InferenceJob({
    required this.imageBytes,
    required this.imageW,
    required this.imageH,
    required this.rotation,
  });
}

// ─── Isolate entry point ──────────────────────────────────────────────────────

/// Long-lived isolate — initialized once, processes jobs until killed.
void _isolateEntry(_IsolateStartup startup) {
  // Set static label list so _parseOutput can use it
  YoloDetector._labelList = startup.labels;

  // Build interpreter from raw bytes — no rootBundle or platform channels
  final opts        = InterpreterOptions()..threads = 2;
  final interpreter = Interpreter.fromBuffer(startup.modelBytes, options: opts);
  interpreter.allocateTensors();

  // Hand our SendPort back to the main isolate so it can send us jobs
  final jobPort = ReceivePort();
  startup.mainSendPort.send(jobPort.sendPort);

  // Process jobs as they arrive
  jobPort.listen((message) {
    if (message is! _InferenceJob) return;
    try {
      final rgb   = YoloDetector._nv21ToRgb(
          message.imageBytes, message.imageW, message.imageH, message.rotation);
      final input = YoloDetector._imageToFloat32(rgb);
      // In _isolateEntry, replace the _runInference call with:
final isRotated = message.rotation == 90 || message.rotation == 270;
final dets = YoloDetector._runInference(
    interpreter, input,
    isRotated ? message.imageH.toDouble() : message.imageW.toDouble(),
    isRotated ? message.imageW.toDouble() : message.imageH.toDouble(),
);

      startup.mainSendPort.send(dets.map((d) => {
        'left':       d.boundingBox.left,
        'top':        d.boundingBox.top,
        'right':      d.boundingBox.right,
        'bottom':     d.boundingBox.bottom,
        'label':      d.label,
        'confidence': d.confidence,
      }).toList());
    } catch (e) {
      startup.mainSendPort.send('error: $e');
    }
  });
}


