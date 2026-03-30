// FILE: lib/providers.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'services/base_detector.dart';
import 'services/base_distance.dart';
import 'services/detection_service.dart';
import 'services/distance_service.dart';
import 'services/yolo_detector.dart';

// ── Active detector ───────────────────────────────────────────────────────────

final activeDetectorTypeProvider =
    StateProvider<DetectorType>((ref) => DetectorType.mlKit);

final activeDetectorProvider =
    StateNotifierProvider<DetectorNotifier, BaseDetector>((ref) {
  return DetectorNotifier(ref);
});

class DetectorNotifier extends StateNotifier<BaseDetector> {
  final Ref _ref;

  DetectorNotifier(this._ref) : super(MlKitDetector()) {
    _ref.listen<DetectorType>(activeDetectorTypeProvider, (prev, next) {
      if (prev != next) _switchTo(next);
    });
    state.initialize();
  }

  Future<void> _switchTo(DetectorType type) async {
    final oldDetector = state;

    final next = _build(type);
    await next.initialize();

    // Swap state first so the UI picks up the new detector immediately,
    // then close the old one in the background.
    state = next;
    await oldDetector.close();
  }

  BaseDetector _build(DetectorType type) {
    switch (type) {
      case DetectorType.mlKit:
        return MlKitDetector();
      case DetectorType.yolo:
        return YoloDetector();
    }
  }
}

// ── Active distance service ───────────────────────────────────────────────────

final activeDistanceTypeProvider =
    StateProvider<DistanceType>((ref) => DistanceType.heuristic);

final activeDistanceProvider = Provider<BaseDistanceService>((ref) {
  final type = ref.watch(activeDistanceTypeProvider);
  switch (type) {
    case DistanceType.heuristic:
      return HeuristicDistanceService();
    case DistanceType.focalLength:
      // Step 3 will fill this in.
      return HeuristicDistanceService();
  }
});
