// FILE: lib/home_screen.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'providers.dart';
import 'services/base_detector.dart';
import 'live_detection_screen.dart';
import 'image_detection_screen.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final activeDetector = ref.watch(activeDetectorTypeProvider);

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF080C14),
              Color(0xFF0A1020),
              Color(0xFF0D1528),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 56),
                _buildHeader(),
                const SizedBox(height: 36),

                // ── Model selector ──────────────────────────────────────────
                _ModelSelector(
                  selected: activeDetector,
                  onChanged: (type) =>
                      ref.read(activeDetectorTypeProvider.notifier).state = type,
                ),

                const SizedBox(height: 32),

                // ── Action cards ────────────────────────────────────────────
                _DetectionCard(
                  icon: Icons.videocam_rounded,
                  title: 'Live Detection',
                  subtitle: 'Real-time object detection\nvia camera stream',
                  accentColor: const Color(0xFF00E5FF),
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => const LiveDetectionScreen(),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                _DetectionCard(
                  icon: Icons.image_search_rounded,
                  title: 'Image Detection',
                  subtitle: 'Upload an image and detect\nobjects with distance',
                  accentColor: const Color(0xFF7C4DFF),
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => const ImageDetectionScreen(),
                    ),
                  ),
                ),

                const Spacer(),
                _buildFooter(),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            border: Border.all(
                color: const Color(0xFF00E5FF).withOpacity(0.5), width: 1),
            borderRadius: BorderRadius.circular(6),
            color: const Color(0xFF00E5FF).withOpacity(0.05),
          ),
          child: const Text(
            'COMPUTER VISION · FLUTTER',
            style: TextStyle(
              color: Color(0xFF00E5FF),
              fontSize: 10,
              letterSpacing: 3,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        const SizedBox(height: 18),
        const Text(
          'Object\nDetector',
          style: TextStyle(
            color: Colors.white,
            fontSize: 46,
            fontWeight: FontWeight.w800,
            height: 1.1,
            letterSpacing: -1.5,
          ),
        ),
        const SizedBox(height: 12),
        Text(
          'Detect objects & estimate distance\nin real-time or from your gallery.',
          style: TextStyle(
            color: Colors.white.withOpacity(0.4),
            fontSize: 14,
            height: 1.6,
          ),
        ),
      ],
    );
  }

  Widget _buildFooter() {
    return Row(
      children: [
        Icon(Icons.school_rounded,
            size: 13, color: Colors.white.withOpacity(0.25)),
        const SizedBox(width: 6),
        Text(
          "Master's Project · Computer Vision",
          style: TextStyle(
            color: Colors.white.withOpacity(0.25),
            fontSize: 11,
            letterSpacing: 0.5,
          ),
        ),
      ],
    );
  }
}

// ─── Model Selector ───────────────────────────────────────────────────────────

class _ModelSelector extends StatelessWidget {
  final DetectorType selected;
  final ValueChanged<DetectorType> onChanged;

  const _ModelSelector({
    required this.selected,
    required this.onChanged,
  });

  // To add a new model in future: just add an entry to this list.
  // No other UI code needs to change.
  static const _models = [
    _ModelOption(
      type: DetectorType.mlKit,
      label: 'ML Kit',
      sublabel: 'Google · Fast',
      icon: Icons.bolt_rounded,
      color: Color(0xFF00E5A0),
    ),
    _ModelOption(
      type: DetectorType.yolo,
      label: 'YOLOv8n',
      sublabel: 'TFLite · Accurate',
      icon: Icons.psychology_rounded,
      color: Color(0xFF00E5FF),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section label
        Padding(
          padding: const EdgeInsets.only(left: 2, bottom: 12),
          child: Row(
            children: [
              Container(
                width: 3,
                height: 12,
                decoration: BoxDecoration(
                  color: const Color(0xFF00E5FF),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 8),
              Text(
                'DETECTION MODEL',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.4),
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2,
                ),
              ),
            ],
          ),
        ),

        // Chip row
        Row(
          children: _models.map((model) {
            final isSelected = selected == model.type;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(
                  right: model == _models.last ? 0 : 10,
                ),
                child: _ModelChip(
                  option: model,
                  isSelected: isSelected,
                  onTap: () => onChanged(model.type),
                ),
              ),
            );
          }).toList(),
        ),

        // Active model description line
        const SizedBox(height: 10),
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 250),
          child: Padding(
            key: ValueKey(selected),
            padding: const EdgeInsets.only(left: 2),
            child: Text(
              _descriptionFor(selected),
              style: TextStyle(
                color: Colors.white.withOpacity(0.3),
                fontSize: 11,
                height: 1.5,
              ),
            ),
          ),
        ),
      ],
    );
  }

  String _descriptionFor(DetectorType type) {
    switch (type) {
      case DetectorType.mlKit:
        return 'Google ML Kit — on-device, no model file required.\nFast inference, good for general use.';
      case DetectorType.yolo:
        return 'YOLOv8n via TFLite — 80 COCO classes.\nHigher accuracy, runs on background isolate.';
    }
  }
}

class _ModelOption {
  final DetectorType type;
  final String label;
  final String sublabel;
  final IconData icon;
  final Color color;

  const _ModelOption({
    required this.type,
    required this.label,
    required this.sublabel,
    required this.icon,
    required this.color,
  });
}

class _ModelChip extends StatelessWidget {
  final _ModelOption option;
  final bool isSelected;
  final VoidCallback onTap;

  const _ModelChip({
    required this.option,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        decoration: BoxDecoration(
          color: isSelected
              ? option.color.withOpacity(0.10)
              : Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected
                ? option.color.withOpacity(0.55)
                : Colors.white.withOpacity(0.08),
            width: 1.5,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: option.color.withOpacity(0.12),
                    blurRadius: 16,
                    spreadRadius: 1,
                  )
                ]
              : null,
        ),
        child: Row(
          children: [
            // Icon container
            AnimatedContainer(
              duration: const Duration(milliseconds: 220),
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: isSelected
                    ? option.color.withOpacity(0.15)
                    : Colors.white.withOpacity(0.06),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                option.icon,
                color: isSelected ? option.color : Colors.white38,
                size: 18,
              ),
            ),
            const SizedBox(width: 10),

            // Labels
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    option.label,
                    style: TextStyle(
                      color: isSelected ? Colors.white : Colors.white54,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    option.sublabel,
                    style: TextStyle(
                      color: isSelected
                          ? option.color.withOpacity(0.7)
                          : Colors.white24,
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),

            // Selected indicator dot
            AnimatedOpacity(
              opacity: isSelected ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 200),
              child: Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: option.color,
                  boxShadow: [
                    BoxShadow(
                      color: option.color.withOpacity(0.6),
                      blurRadius: 6,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Detection Card ───────────────────────────────────────────────────────────

class _DetectionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color accentColor;
  final VoidCallback onTap;

  const _DetectionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.accentColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(22),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: accentColor.withOpacity(0.25),
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: accentColor.withOpacity(0.06),
              blurRadius: 20,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: accentColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: accentColor, size: 26),
            ),
            const SizedBox(width: 18),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.4),
                      fontSize: 12,
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            Container(
              width: 30,
              height: 30,
              decoration: BoxDecoration(
                color: accentColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                Icons.arrow_forward_ios_rounded,
                color: accentColor.withOpacity(0.8),
                size: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}