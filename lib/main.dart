import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: RealtimeClassificationScreen(cameras: cameras),
    );
  }
}

class RealtimeClassificationScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const RealtimeClassificationScreen({super.key, required this.cameras});

  @override
  State<RealtimeClassificationScreen> createState() => _RealtimeClassificationScreenState();
}

class _RealtimeClassificationScreenState extends State<RealtimeClassificationScreen> {
  late CameraController _cameraController;
  String _result = '分類中...';
  Timer? _predictTimer;
  OrtSession? _session;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameraController = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await _cameraController.initialize();
    await _cameraController.setFlashMode(FlashMode.off);
    await _loadModel();
    _startPredictionLoop();
    setState(() {});
  }

  Future<void> _loadModel() async {
    OrtEnv.instance.init();
    final modelData = await rootBundle.load('assets/models/coin_model_v2.onnx');
    _session = OrtSession.fromBuffer(modelData.buffer.asUint8List(), OrtSessionOptions());
  }

  Future<List<String>> _loadLabels() async {
    final labelString = await rootBundle.loadString('assets/labels.txt');
    return labelString.split('\n').map((e) => e.trim()).toList();
  }

  void _startPredictionLoop() {
    _predictTimer = Timer.periodic(const Duration(milliseconds: 1000), (_) async {
      if (!_cameraController.value.isInitialized || _session == null) return;

      try {
        final file = await _cameraController.takePicture();
        final image = img.decodeImage(await file.readAsBytes())!;
        final resized = img.copyResize(image, width: 512, height: 512);

        final input = Float32List(1 * 3 * 512 * 512);
        int index = 0;
        for (int c = 0; c < 3; c++) {
          for (int y = 0; y < 512; y++) {
            for (int x = 0; x < 512; x++) {
              final pixel = resized.getPixel(x, y);
              final value = ((c == 0 ? img.getRed(pixel) : c == 1 ? img.getGreen(pixel) : img.getBlue(pixel)) / 255.0);
              input[index++] = value;
            }
          }
        }

        final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, 512, 512]);
        final outputs = _session!.run(OrtRunOptions(), {'data': inputTensor});
        final output = outputs[0]?.value as List<List<double>>;
        final flat = output.expand((e) => e).toList();

        final labels = await _loadLabels();
        final Map<String, int> coinValues = {
          '1yen': 1,
          '5yen': 5,
          '10yen': 10,
          '50yen': 50,
          '100yen': 100,
          '500yen': 500,
          'other': 0,
        };

        final labelWithCount = <String>[];
        int totalYen = 0;

        for (int i = 0; i < flat.length; i++) {
          final count = flat[i].round();
          final rawLabel = labels[i];
          final normalizedLabel = rawLabel.toLowerCase().replaceAll(' ', '');

          if (count > 0 && normalizedLabel != 'other') {
            final value = coinValues[normalizedLabel] ?? 0;
            labelWithCount.add('$rawLabel $count');
            totalYen += value * count;
          }
        }

        labelWithCount.add('合計: $totalYen円');

        setState(() {
          _result = labelWithCount.join('\n');
        });

        inputTensor.release();
        outputs.forEach((e) => e?.release());
      } catch (e) {
        debugPrint('予測中のエラー: $e');
      }
    });
  }

  @override
  void dispose() {
    _predictTimer?.cancel();
    _cameraController.dispose();
    _session?.release();
    OrtEnv.instance.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController),
          Container(
            alignment: Alignment.bottomCenter,
            margin: const EdgeInsets.all(32),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                _result,
                style: const TextStyle(color: Colors.white, fontSize: 18),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
