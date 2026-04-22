// FILE: lib/services/midas_isolate.dart
import 'dart:isolate';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

const int kMidasSize = 256;

/// Message sent BACK from the isolate — this direction works fine.
class MidasResult {
  final Float32List depthMap;
  MidasResult(this.depthMap);
}

Future<SendPort> spawnMidasIsolate() async {
  final receivePort = ReceivePort();
  await Isolate.spawn(_midasIsolateMain, receivePort.sendPort);
  return await receivePort.first as SendPort;
}

void _midasIsolateMain(SendPort callerPort) async {
  final receivePort = ReceivePort();
  callerPort.send(receivePort.sendPort);

  Interpreter? interpreter;
  bool modelLoaded = false;

  await for (final msg in receivePort) {
    // First message is always the model bytes
    if (!modelLoaded) {
      try {
        final modelBytes = msg as Uint8List;
        print('🟡 MiDaS isolate: got model bytes — ${modelBytes.lengthInBytes}');
        interpreter = Interpreter.fromBuffer(modelBytes);
        modelLoaded = true;
        print('🟡 MiDaS isolate: interpreter created ✅');
      } catch (e) {
        print('❌ MiDaS isolate: setup failed — $e');
      }
      continue; // wait for next message (first real frame)
    }

    // Subsequent messages are inference requests
    print('🟣 MiDaS isolate: received message type=${msg.runtimeType}');
    if (msg is Map && interpreter != null) {
      final yuvBytes = msg['yuvBytes'] as Uint8List;
      final imageW   = msg['imageW']   as int;
      final imageH   = msg['imageH']   as int;
      final rotation = msg['rotation'] as int;
      final replyTo  = msg['replyTo']  as SendPort;
      try {
        print('🟣 MiDaS isolate: dispatching to _runInference...');
        final depth = _runInference(interpreter, yuvBytes, imageW, imageH, rotation);
        replyTo.send(MidasResult(depth));
        print('🟣 MiDaS isolate: reply sent ✅');
      } catch (e, st) {
        print('❌ MiDaS isolate: _runInference crashed — $e\n$st');
        replyTo.send(null);
      }
    }
  }
}

Float32List _runInference(
  Interpreter interp,
  Uint8List yuvBytes,
  int imageW,
  int imageH,
  int rotation,
) {
  print('🟣 MiDaS _runInference: start');

  // Real path: decode YUV → resize → normalise
  final rgbImage = _yuvToRgb(yuvBytes, imageW, imageH, rotation);
  final resized = img.copyResize(rgbImage, width: kMidasSize, height: kMidasSize);

  final input = List.generate(
    1,
    (_) => List.generate(
      kMidasSize,
      (y) => List.generate(
        kMidasSize,
        (x) {
          final pixel = resized.getPixel(x, y);
          return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
        },
      ),
    ),
  );

  final output = List.generate(
    1,
    (_) => List.generate(kMidasSize, (_) => List.generate(kMidasSize, (_) => [0.0])),
  );

  print('🟣 MiDaS _runInference: calling interp.run...');
  interp.run(input, output);
  print('🟣 MiDaS _runInference: interp.run complete ✅');

  // Flatten, filter NaN/Inf, then normalise
  final flat = Float32List(kMidasSize * kMidasSize);
  double minV = double.maxFinite, maxV = -double.maxFinite;
  int i = 0;
  for (final row in output[0]) {
    for (final col in row) {
      double v = col[0];
      // Replace any bad values with 0 before computing range
      if (v.isNaN || v.isInfinite) v = 0.0;
      flat[i++] = v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }

  // Safety check — if the whole map is bad, return zeros
  if (minV == maxV || minV == double.maxFinite) {
    print('⚠️ MiDaS: depth map all same value or all bad — returning zeros');
    return Float32List(kMidasSize * kMidasSize);
  }

  final range = maxV - minV;
  for (int j = 0; j < flat.length; j++) {
    flat[j] = (flat[j] - minV) / range; // [0,1], no inversion
  }
  return flat;
}

img.Image _yuvToRgb(Uint8List bytes, int w, int h, int rotation) {
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
  switch (rotation) {
    case 90:  return img.copyRotate(rgb, angle: 90);
    case 180: return img.copyRotate(rgb, angle: 180);
    case 270: return img.copyRotate(rgb, angle: 270);
    default:  return rgb;
  }
}