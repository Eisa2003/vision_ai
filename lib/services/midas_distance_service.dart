// FILE: lib/services/midas_distance_service.dart
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/painting.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'base_distance.dart';
import 'midas_isolate.dart';

class MidasDistanceService extends BaseDistanceService {
  SendPort? _isolateSendPort;
  bool _inferenceInFlight = false;
  Float32List? _depthMap;
  ReceivePort? _replyPort;

  @override
  String get name => 'MiDaS depth';

  @override
  String get description => 'Neural depth map (MiDaS v2.1 small)';

  Future<void> init() async {
    print('🟡 MiDaS init: loading model bytes...');
    final modelBytes = await rootBundle
        .load('assets/models/midas_v21_small.tflite')
        .then((bd) => bd.buffer.asUint8List());
    print('🟡 MiDaS init: model bytes loaded — ${modelBytes.lengthInBytes} bytes');

    _isolateSendPort = await spawnMidasIsolate();
    print('🟡 MiDaS init: isolate spawned, sending model bytes...');
    _isolateSendPort!.send(modelBytes);
    print('🟡 MiDaS init: done ✅');
  }

  void submitFrame({
    required Uint8List yuvBytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) {
    print('🟡 MiDaS submitFrame: inFlight=$_inferenceInFlight port=${_isolateSendPort != null}');
    if (_inferenceInFlight || _isolateSendPort == null) return;
    _inferenceInFlight = true;

    _replyPort?.close();
    _replyPort = ReceivePort();

    _replyPort!.listen((msg) {
      _replyPort!.close();
      _replyPort = null;
      _inferenceInFlight = false;
      if (msg is MidasResult) {
        _depthMap = msg.depthMap;
        final mid = msg.depthMap[kMidasSize * kMidasSize ~/ 2];
        print('✅ MiDaS: depth map received | centre=$mid');
      } else {
        print('❌ MiDaS: isolate returned null');
      }
    }); // ← _replyPort!.listen closes here

    Future.delayed(const Duration(seconds: 30), () {
      if (_inferenceInFlight) {
        print('⚠️ MiDaS: timeout — resetting inFlight flag');
        _inferenceInFlight = false;
        _replyPort?.close();
        _replyPort = null;
      }
    }); // ← Future.delayed closes here

    // Outside both callbacks — sends immediately
    _isolateSendPort!.send({
      'yuvBytes': yuvBytes,
      'imageW':   imageW,
      'imageH':   imageH,
      'rotation': rotation,
      'replyTo':  _replyPort!.sendPort,
    });
  } // ← submitFrame closes here

 @override
double estimateMeters(Rect boundingBox, Size imageSize) {
  final map = _depthMap;
  if (map == null) {
    print('🔵 MiDaS: depthMap is null — isolate not yet replied');
    return -1;
  }

  final cx = ((boundingBox.center.dx / imageSize.width) * kMidasSize)
      .clamp(0, kMidasSize - 1).toInt();
  final cy = ((boundingBox.center.dy / imageSize.height) * kMidasSize)
      .clamp(0, kMidasSize - 1).toInt();

  double sum = 0;
  int count = 0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      final nx = (cx + dx).clamp(0, kMidasSize - 1);
      final ny = (cy + dy).clamp(0, kMidasSize - 1);
      sum += map[ny * kMidasSize + nx];
      count++;
    }
  }

  final normalised = sum / count;
print('🔵 MiDaS: cx=$cx cy=$cy | normalised=$normalised | sum=$sum count=$count');

if (normalised.isNaN || normalised.isInfinite) {
  print('❌ MiDaS: bad normalised value, returning -1');
  return -1;
}

  // MiDaS disparity: high value = close object
  // Convert: meters = scale / (normalised + epsilon)
  // At normalised=1.0 (touching) → ~_scale meters... that's wrong.
  // We want: normalised=1.0 → ~0.3m, normalised=0.1 → ~3m
  const double _near = 0.3;  // closest readable distance (metres)
  const double _far  = 8.0;  // furthest readable distance (metres)
  final meters = _near + (_far - _near) * (1.0 - normalised);

  print('🔵 MiDaS: meters=${meters.toStringAsFixed(2)}');
  return meters;
}

  @override
String estimate(Rect boundingBox, Size imageSize) {
  final m = estimateMeters(boundingBox, imageSize);
  if (m < 0 || m.isNaN || m.isInfinite) return '—';
  if (m < 0.5) return '< 0.5 m';
  return '~${m.toStringAsFixed(1)} m';
}
}