// FILE: lib/services/midas_isolate.dart
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

const int kMidasSize = 256;

/// Message sent TO the isolate per frame.
class MidasRequest {
  final Uint8List yuvBytes;  // raw NV21/YUV420 plane bytes
  final int imageW;
  final int imageH;
  final int rotation;        // sensor orientation degrees
  final SendPort replyTo;

  MidasRequest({
    required this.yuvBytes,
    required this.imageW,
    required this.imageH,
    required this.rotation,
    required this.replyTo,
  });
}

/// Message sent BACK from the isolate.
class MidasResult {
  /// Flat float32 array, length 256*256.
  /// Values are INVERTED so that higher == farther (meters-friendly).
  final Float32List depthMap;
  MidasResult(this.depthMap);
}

/// Entry-point for the long-lived background isolate.
/// Call once at app start, keep the SendPort.
Future<SendPort> spawnMidasIsolate() async {
  final receivePort = ReceivePort();
  await Isolate.spawn(_midasIsolateMain, receivePort.sendPort);
  return await receivePort.first as SendPort;
}

/// Runs inside the isolate — loads the interpreter once, then loops.
void _midasIsolateMain(SendPort callerPort) async {
  final receivePort = ReceivePort();
  callerPort.send(receivePort.sendPort);

  Uint8List modelBytes;
  try {
    modelBytes = await receivePort.first as Uint8List;
    print('🟡 MiDaS isolate: got model bytes — ${modelBytes.lengthInBytes}');
  } catch (e) {
    print('❌ MiDaS isolate: failed to receive model bytes — $e');
    return;
  }

  Interpreter interpreter;
  try {
    interpreter = Interpreter.fromBuffer(modelBytes);
    print('🟡 MiDaS isolate: interpreter created ✅');
  } catch (e) {
    print('❌ MiDaS isolate: failed to create interpreter — $e');
    return;
  }

  await for (final msg in receivePort) {
    if (msg is MidasRequest) {
      try {
        final depth = _runInference(interpreter, msg);
        msg.replyTo.send(MidasResult(depth));
        print('🟡 MiDaS isolate: inference done, reply sent');
      } catch (e, st) {
        print('❌ MiDaS isolate: inference failed — $e\n$st');
        msg.replyTo.send(null);
      }
    }
  }
}

Float32List _runInference(Interpreter interp, MidasRequest req) {
  // 1. Decode YUV → RGB image
  final rgbImage = _yuvToRgb(req.yuvBytes, req.imageW, req.imageH, req.rotation);

  // 2. Resize to 256×256
  final resized = img.copyResize(rgbImage, width: kMidasSize, height: kMidasSize);

  // 3. Build input tensor [1, 256, 256, 3] normalised to [0,1]
  final input = List.generate(
    1,
    (_) => List.generate(
      kMidasSize,
      (y) => List.generate(
        kMidasSize,
        (x) {
          final pixel = resized.getPixel(x, y);
          return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0,
          ];
        },
      ),
    ),
  );

  // 4. Output tensor [1, 256, 256, 1]
  final output = List.generate(
    1,
    (_) => List.generate(
      kMidasSize,
      (_) => List.generate(kMidasSize, (_) => [0.0]),
    ),
  );

  interp.run(input, output);

  // 5. Flatten + invert (MiDaS gives inverse depth; invert so high == far)
  final flat = Float32List(kMidasSize * kMidasSize);
  double minV = double.maxFinite, maxV = -double.maxFinite;
  int i = 0;
  for (final row in output[0]) {
    for (final col in row) {
      final v = col[0];
      flat[i++] = v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }

  // Normalise to [0,1] then invert: 1 = far, 0 = close
  final range = (maxV - minV).clamp(1e-6, double.maxFinite);
  for (int j = 0; j < flat.length; j++) {
    flat[j] = 1.0 - ((flat[j] - minV) / range);
  }
  return flat;
}

img.Image _yuvToRgb(Uint8List bytes, int w, int h, int rotation) {
  // Simple NV21 → RGB conversion (same approach as your YOLO isolate)
  final rgb = img.Image(width: w, height: h);
  final uvStart = w * h;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final yVal = bytes[y * w + x];
      final uvIndex = uvStart + (y ~/ 2) * w + (x & ~1);
      final v = bytes[uvIndex] - 128;
      final u = bytes[uvIndex + 1] - 128;
      final r = (yVal + 1.402 * v).clamp(0, 255).toInt();
      final g = (yVal - 0.344136 * u - 0.714136 * v).clamp(0, 255).toInt();
      final b = (yVal + 1.772 * u).clamp(0, 255).toInt();
      rgb.setPixelRgb(x, y, r, g, b);
    }
  }
  // Rotate to match display orientation
  switch (rotation) {
    case 90:  return img.copyRotate(rgb, angle: 90);
    case 180: return img.copyRotate(rgb, angle: 180);
    case 270: return img.copyRotate(rgb, angle: 270);
    default:  return rgb;
  }
}