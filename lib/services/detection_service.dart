// FILE: lib/services/detection_service.dart

import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'base_detector.dart';

/// ML Kit implementation of [BaseDetector].
/// Behaviour is identical to before — it now just conforms to the shared interface.
class MlKitDetector extends BaseDetector {
  ObjectDetector? _detector;

  @override
  String get name => 'ML Kit';

  @override
  String get description => 'Google ML Kit · broad categories · fastest';

  @override
  Future<void> initialize() async {
    final options = ObjectDetectorOptions(
      mode: DetectionMode.stream,
      classifyObjects: true,
      multipleObjects: true,
    );
    _detector = ObjectDetector(options: options);
  }

  @override
  Future<List<Detection>> detect(InputImage image) async {
    if (_detector == null) return [];
    try {
      final objects = await _detector!.processImage(image);
      return objects.map((obj) {
        final label = obj.labels.isEmpty ? 'Unknown' : obj.labels.first.text;
        final confidence =
            obj.labels.isEmpty ? 0.0 : obj.labels.first.confidence;
        return Detection(
          boundingBox: obj.boundingBox,
          label: label,
          confidence: confidence,
        );
      }).toList();
    } catch (_) {
      return [];
    }
  }

  @override
  Future<void> close() async {
    await _detector?.close();
    _detector = null;
  }

  /// Convenience: detect from a file path (used by ImageDetectionScreen).
  Future<List<Detection>> detectFile(String filePath) =>
      detect(InputImage.fromFilePath(filePath));
}
