package com.chili.gumamelaclassifier;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.chili.gumamelaclassifier.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView, loadingImageView;
    Button picture, selectImageButton;
    Animation clickAnimation;
    Model model; // Load the model once for reuse
    private final int DELAY_MILLIS = 3000;  // Delay time in milliseconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize Views
        initViews();

        // Initialize Model
        initModel();

        // Set button listeners
        setButtonListeners();
    }

    private void initViews() {
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        loadingImageView = findViewById(R.id.loadingImageView);
        picture = findViewById(R.id.button);
        selectImageButton = findViewById(R.id.button1);
        clickAnimation = AnimationUtils.loadAnimation(this, R.anim.button_click_animation);
    }

    private void initModel() {
        try {
            // Load the model only once to optimize performance
            model = Model.newInstance(this);
        } catch (IOException e) {
            e.printStackTrace();
            // Add user feedback if the model fails to load
            result.setText("Error loading model.");
        }
    }

    private void setButtonListeners() {
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                applyClickAnimation(view);
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                } else {
                    openCamera();
                }
            }
        });

        selectImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                applyClickAnimation(v);
                openGallery();
            }
        });
    }

    private void applyClickAnimation(View view) {
        view.startAnimation(clickAnimation);
    }

    private void openGallery() {
        resetUI();
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, 2);
    }

    private void openCamera() {
        resetUI();
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, 1);
    }

    private void resetUI() {
        // Clear previous results and reset UI
        result.setText("");
        confidence.setText("");
        imageView.setImageResource(R.drawable.over);  // Reset to placeholder image
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Bitmap imageBitmap = null;
            if (requestCode == 1) {  // From Camera
                Bundle extras = data.getExtras();
                if (extras != null) {
                    imageBitmap = (Bitmap) extras.get("data");
                }
            } else if (requestCode == 2) {  // From Gallery
                try {
                    imageBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(data.getData()));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (imageBitmap != null) {
                handleImage(imageBitmap);
            }
        }
    }

    private void handleImage(Bitmap imageBitmap) {
        resetUI();

        int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
        imageBitmap = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension);
        imageView.setImageBitmap(imageBitmap);

        new ClassifyTask().execute(imageBitmap);  // Run classification on a background thread
    }

    private class ClassifyTask extends AsyncTask<Bitmap, Void, String[]> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            showLoadingAnimation();
        }

        @Override
        protected String[] doInBackground(Bitmap... bitmaps) {
            try {
                // Simulate processing delay
                Thread.sleep(DELAY_MILLIS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            return classifyImage(bitmaps[0]);
        }

        @Override
        protected void onPostExecute(String[] classificationResults) {
            super.onPostExecute(classificationResults);
            hideLoadingAnimation();

            if (classificationResults != null && classificationResults.length > 0) {
                result.setText(classificationResults[0]);
                confidence.setText(classificationResults[1]);
            } else {
                result.setText("Classification failed.");
            }
        }
    }

    private String[] classifyImage(Bitmap image) {
        String[][] classes = {
                {"Malvacea", "Hibiscus", "Hibiscus rosa-sinensis", "Red Cluster Hibiscus"},
                {"Malvacea", "Hibiscus", "Hibiscus arnottianus", "Hawaiian White"},
                {"Malvacea", "Hibiscus", "Hibiscus rosa-sinensis", "Pink Chinese Hibiscus"},
                {"Rosaceae", "Rosa", "Rosa", "Mister Lincoln"},
                {"Rosaceae", "Rosa", "Rosa", "Maskara"},
                {"Rosaceae", "Rosa", "Rosa", "Iceberg"},
                {"Unknown", "Unknown", "Unknown", "Unknown Flower"}
        };

        try {
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, 224, 224, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[224 * 224];
            resizedImage.getPixels(intValues, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());

            for (int pixel = 0; pixel < intValues.length; pixel++) {
                int val = intValues[pixel];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));  // Red
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));   // Green
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));          // Blue
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Run inference
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences = outputFeature0.getFloatArray();

            // Get the classification result
            int maxPos = 0;
            float maxConfidence = confidences[0];
            for (int i = 1; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String resultText = String.format(
                    "\nFamily: %s\nGenus: %s\nSpecies: %s\nCommon Name: %s\n",
                    classes[maxPos][0], classes[maxPos][1], classes[maxPos][2], classes[maxPos][3]
            );

            StringBuilder confidenceText = new StringBuilder();
            for (int i = 0; i < confidences.length; i++) {
                confidenceText.append(String.format("%s: %.1f%%\n", classes[i][3], confidences[i] * 100));
            }

            saveClassificationResults(confidences, maxPos);

            return new String[]{resultText, confidenceText.toString()};

        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    public void showLoadingAnimation() {
        Button takeButton = findViewById(R.id.button);
        Button selectButton = findViewById(R.id.button1);
        takeButton.setVisibility(View.INVISIBLE);
        selectButton.setVisibility(View.INVISIBLE);

        View overlayView = findViewById(R.id.overlayView);
        ImageView loadingImageView = findViewById(R.id.loadingImageView);
        overlayView.setVisibility(View.VISIBLE);
        loadingImageView.setVisibility(View.VISIBLE);
        Glide.with(this).asGif().load(R.drawable.load).into(loadingImageView);
    }

    public void hideLoadingAnimation() {
        Button takeButton = findViewById(R.id.button);
        Button selectButton = findViewById(R.id.button1);
        takeButton.setVisibility(View.VISIBLE);
        selectButton.setVisibility(View.VISIBLE);

        View overlayView = findViewById(R.id.overlayView);
        ImageView loadingImageView = findViewById(R.id.loadingImageView);
        overlayView.setVisibility(View.GONE);
        loadingImageView.setVisibility(View.GONE);
    }

    private void saveClassificationResults(float[] confidences, int predictedClass) {
        SharedPreferences sharedPreferences = getSharedPreferences("ClassificationHistory", MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();

        long timestamp = System.currentTimeMillis();
        editor.putLong(String.valueOf(timestamp), timestamp);

        editor.putInt(timestamp + "_PredictedClass", predictedClass);

        String classKey = "ClassCount_" + predictedClass;
        int currentCount = sharedPreferences.getInt(classKey, 0);
        editor.putInt(classKey, currentCount + 1);

        editor.apply();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Clean up the model when the activity is destroyed
        if (model != null) {
            model.close();
        }
    }
}
