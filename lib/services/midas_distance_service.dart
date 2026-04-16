// FILE: lib/services/midas_distance_service.dart
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/painting.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'base_distance.dart';
import 'midas_isolate.dart';

/// Calibration: maps normalised depth [0,1] → estimated metres.
/// Tune [_scale] against a real ruler for your camera.
const double _scale = 10.0; // 0.0 (far) * 10 = 0 m  →  1.0 (close) * 10 = 10 m

class MidasDistanceService extends BaseDistanceService {
  SendPort? _isolateSendPort;
  bool _inferenceInFlight = false;

  /// Latest depth map from the isolate — 256×256 floats, [0,1], 1=close.
  Float32List? _depthMap;

  @override
  String get name => 'MiDaS depth';

  @override
  String get description => 'Neural depth map (MiDaS v2.1 small)';

  /// Call once (e.g. in your Riverpod provider initialiser).
  Future<void> init() async {
    print('🟡 MiDaS init: loading model bytes...');
    final modelBytes = await rootBundle
        .load('assets/models/midas_v21_small.tflite')
        .then((bd) => bd.buffer.asUint8List());
    print(
        '🟡 MiDaS init: model bytes loaded — ${modelBytes.lengthInBytes} bytes');

    _isolateSendPort = await spawnMidasIsolate();
    print('🟡 MiDaS init: isolate spawned, sending model bytes...');
    _isolateSendPort!.send(modelBytes);
    print('🟡 MiDaS init: done ✅');
  }

  /// Called from _onCameraFrame — same fire-and-forget pattern as YOLO.
  /// Returns immediately; depth map is updated when the isolate replies.
  void submitFrame({
    required Uint8List yuvBytes,
    required int imageW,
    required int imageH,
    required int rotation,
  }) {
    print('🟡 MiDaS submitFrame: inFlight=$_inferenceInFlight port=${_isolateSendPort != null}');
    if (_inferenceInFlight || _isolateSendPort == null) return;
    _inferenceInFlight = true;

    final replyPort = ReceivePort();
    replyPort.listen((msg) {
      replyPort.close();
      _inferenceInFlight = false;
      if (msg is MidasResult) {
        _depthMap = msg.depthMap;
        // Sample the centre pixel to confirm values look sane
        final mid = msg.depthMap[kMidasSize * kMidasSize ~/ 2];
        print(
            '✅ MiDaS: depth map received | centre=$mid | min=${msg.depthMap.reduce((a, b) => a < b ? a : b).toStringAsFixed(3)} max=${msg.depthMap.reduce((a, b) => a > b ? a : b).toStringAsFixed(3)}');
      } else {
        print('❌ MiDaS: isolate returned null — inference threw an exception');
      }
    });

    _isolateSendPort!.send(MidasRequest(
      yuvBytes: yuvBytes,
      imageW: imageW,
      imageH: imageH,
      rotation: rotation,
      replyTo: replyPort.sendPort,
    ));
  }

  @override
  double estimateMeters(Rect boundingBox, Size imageSize) {
    final map = _depthMap;
    if (map == null) {
      print('🔵 MiDaS: depthMap is null — isolate not yet replied');
      return -1;
    }

    // Sample the depth at the bounding-box centre, mapped to 256×256
    final cx = ((boundingBox.center.dx / imageSize.width) * kMidasSize)
        .clamp(0, kMidasSize - 1)
        .toInt();
    final cy = ((boundingBox.center.dy / imageSize.height) * kMidasSize)
        .clamp(0, kMidasSize - 1)
        .toInt();

    // Average a 5×5 patch for stability
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

    final raw = sum / count;
    final meters = raw * _scale;

    print(
        '🔵 MiDaS: box=${boundingBox.center} → cx=$cx cy=$cy | raw=$raw | meters=${meters.toStringAsFixed(2)}');

    return meters;
  }

  @override
  String estimate(Rect boundingBox, Size imageSize) {
    final m = estimateMeters(boundingBox, imageSize);
    if (m < 0) return '—';
    if (m < 0.5) return '< 0.5 m';
    return '~${m.toStringAsFixed(1)} m';
  }
}
