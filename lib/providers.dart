// FILE: lib/providers.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'services/base_detector.dart';
import 'services/base_distance.dart';
import 'services/detection_service.dart';
import 'services/distance_service.dart';
import 'services/yolo_detector.dart';
import 'services/focal_distance_service.dart';

// ── Active detector ───────────────────────────────────────────────────────────

final activeDetectorTypeProvider =
    StateProvider<DetectorType>((ref) => DetectorType.mlKit);

// This provider holds the currently active detector instance, which the UI listens to for changes.
// We call the BaseDetector to abstract away the specific implementation (ML Kit, YOLO, etc.) from the UI.
// this activeDetectorProvider is a StateNotifierProvider, which means it holds a state (the current detector/BaseDetector type) and allows us to update that state (switching detectors) while notifying the UI of changes.


// Change StateNotifierProvider to AsyncNotifierProvider
final activeDetectorProvider = AsyncNotifierProvider<DetectorNotifier, BaseDetector>(() {
  return DetectorNotifier();
});

class DetectorNotifier extends AsyncNotifier<BaseDetector> {
  @override
  Future<BaseDetector> build() async {
    // 1. Watch the type (YOLO or ML Kit)
    final type = ref.watch(activeDetectorTypeProvider);
    
    // 2. Build the new detector
    final detector = (type == DetectorType.yolo) ? YoloDetector() : MlKitDetector();
    
    // 3. Await the actual heavy loading (Riverpod handles the 'Loading' state here)
    await detector.initialize();
    
    return detector;
  }
}


// ── Active distance service ───────────────────────────────────────────────────

final activeDistanceTypeProvider =
    StateProvider<DistanceType>((ref) => DistanceType.focalLength);

final activeDistanceProvider = Provider<BaseDistanceService>((ref) {
  final type = ref.watch(activeDistanceTypeProvider);
  switch (type) {
    case DistanceType.heuristic:
      return HeuristicDistanceService();
    case DistanceType.focalLength:
      return FocalLengthDistanceService();
  }
});
