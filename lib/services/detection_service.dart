import 'dart:ui';

import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'base_detector.dart';

Rect rotateBox90(Rect box, Size imageSize) {
  final imageWidth = imageSize.width;

  return Rect.fromLTRB(
    box.top,
    imageWidth - box.right,
    box.bottom,
    imageWidth - box.left,
  );
}

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
      // 📏 IMAGE METADATA DEBUG
      final metadata = image.metadata;
      final imageSize = metadata?.size;

      print('================ ML KIT DEBUG ================');
      print('Image size (ML Kit): $imageSize');
      print('Rotation: ${metadata?.rotation}');
      print('Format: ${metadata?.format}');
      print('=============================================');

      final objects = await _detector!.processImage(image);

      return objects.map((obj) {
        final box = obj.boundingBox;

        // 📦 BOX DEBUG
        final width = box.right - box.left;
        final height = box.bottom - box.top;

        print('--- OBJECT DETECTED ---');
        print('Raw box: $box');
        print('Width: $width, Height: $height');

        // 📊 NORMALIZED VALUES (0 → 1 range)
        if (imageSize != null) {
          final normLeft = box.left / imageSize.width;
          final normTop = box.top / imageSize.height;
          final normRight = box.right / imageSize.width;
          final normBottom = box.bottom / imageSize.height;

          print('Normalized: '
              'L=$normLeft, T=$normTop, R=$normRight, B=$normBottom');
        } else {
          print('⚠️ No image size available (metadata null)');
        }

        final label =
            obj.labels.isEmpty ? 'Unknown' : obj.labels.first.text;
        final confidence =
            obj.labels.isEmpty ? 0.0 : obj.labels.first.confidence;

        print('Label: $label (${(confidence * 100).toStringAsFixed(1)}%)');
        print('--------------------------');

        return Detection(
          boundingBox: box, // 🚫 NO OFFSET — raw data only
          label: label,
          confidence: confidence,
        );
      }).toList();
    } catch (e) {
      print('❌ Detection error: $e');
      return [];
    }
  }

  @override
  Future<void> close() async {
    await _detector?.close();
    _detector = null;
  }

  Future<List<Detection>> detectFile(String filePath) =>
      detect(InputImage.fromFilePath(filePath));
}