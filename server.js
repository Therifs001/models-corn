const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');

// Inisialisasi aplikasi Express
const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Simpan model di dalam memori untuk efisiensi
let model;
const loadModel = async () => {
    if (!model) {
        model = await tf.loadLayersModel('model_js/model.json');
    }
};

// Setup untuk upload file menggunakan Multer
const upload = multer({ dest: 'uploads/' });

// Kategori dan penjelasan klasifikasi
const categories = ["Blight", "Common", "Gray", "Healthy"];
const explanations = {
    "Blight": "Blight is a disease caused by fungal or bacterial infections, which can lead to serious damage to the leaves and plants of corn.",
    "Common": "Common refers to the condition of a corn plant that shows no clear signs of disease or visible disturbances.",
    "Gray": "Gray indicates a possible infection or mild damage to the corn plant, which could be caused by environmental factors or a mild disease.",
    "Healthy": "Healthy means the corn plant is in good condition with no signs of issues or disease."
};

// Endpoint untuk klasifikasi gambar
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        await loadModel();

        const filePath = req.file.path;
        const imageBuffer = fs.readFileSync(filePath);
        const tensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims(0);

        const prediction = await model.predict(tensor).data();
        const predictionIndex = prediction.indexOf(Math.max(...prediction));
        const predictedClass = categories[predictionIndex];
        const explanation = explanations[predictedClass];

        // Hapus file setelah diproses
        fs.unlinkSync(filePath);

        res.json({
            success: true,
            prediction: predictedClass,
            explanation: explanation
        });
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).json({ success: false, message: 'Error during prediction', error: error.message });
    }
});

// Endpoint untuk menerima pertanyaan
app.post('/ask', (req, res) => {
    const { question } = req.body;

    if (!question || question.trim() === '') {
        return res.status(400).json({ success: false, message: 'Pertanyaan tidak boleh kosong!' });
    }

    // Dummy respons
    const answer = {
        answer: {
            parts: [{ text: `Ini adalah jawaban untuk pertanyaan: "${question}"` }]
        }
    };

    res.json({ success: true, answer });
});

// Jalankan server
const PORT = 8081;
app.listen(PORT, () => {
    console.log(`Server berjalan di http://127.0.0.1:${PORT}`);
});
