// FILE: lib/home_screen.dart

import 'package:flutter/material.dart';
import 'live_detection_screen.dart';
import 'image_detection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0A0E1A),
              Color(0xFF0D1B2A),
              Color(0xFF112233),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 60),
                _buildHeader(),
                const SizedBox(height: 60),
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
                const SizedBox(height: 20),
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
            border: Border.all(color: const Color(0xFF00E5FF), width: 1),
            borderRadius: BorderRadius.circular(4),
          ),
          child: const Text(
            'ML KIT · FLUTTER',
            style: TextStyle(
              color: Color(0xFF00E5FF),
              fontSize: 11,
              letterSpacing: 3,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: 16),
        const Text(
          'Object\nDetector',
          style: TextStyle(
            color: Colors.white,
            fontSize: 48,
            fontWeight: FontWeight.w800,
            height: 1.1,
            letterSpacing: -1,
          ),
        ),
        const SizedBox(height: 12),
        Text(
          'Detect objects & estimate distance\nin real-time or from your gallery.',
          style: TextStyle(
            color: Colors.white.withOpacity(0.5),
            fontSize: 14,
            height: 1.5,
          ),
        ),
      ],
    );
  }

  Widget _buildFooter() {
    return Row(
      children: [
        Icon(Icons.school_rounded, size: 14, color: Colors.white.withOpacity(0.3)),
        const SizedBox(width: 6),
        Text(
          "Master's Project · Computer Vision",
          style: TextStyle(
            color: Colors.white.withOpacity(0.3),
            fontSize: 12,
            letterSpacing: 0.5,
          ),
        ),
      ],
    );
  }
}

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
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: accentColor.withOpacity(0.3),
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: accentColor.withOpacity(0.08),
              blurRadius: 20,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: accentColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: accentColor, size: 28),
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.5),
                      fontSize: 13,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios_rounded,
              color: accentColor.withOpacity(0.7),
              size: 16,
            ),
          ],
        ),
      ),
    );
  }
}
