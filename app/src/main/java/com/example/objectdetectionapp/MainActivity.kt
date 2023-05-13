package com.example.objectdetectionapp

import android.content.Intent
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import com.example.objectdetectionapp.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp


class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var imageView: ImageView
    lateinit var imageButton: Button
    lateinit var bitmap: Bitmap
    lateinit var model: SsdMobilenetV11Metadata1
    lateinit var labels : List<String>

    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300,300,ResizeOp.ResizeMethod.BILINEAR)).build()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)
        imageView = findViewById(R.id.imageView)
        imageButton = findViewById(R.id.imageButton)

        imageButton.setOnClickListener{
        startActivityForResult(intent, 101)

        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 101){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Releases model resources if no longer used.
        model.close()
    }

    fun get_predictions(){
        // Creates inputs for reference.
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        // Runs model inference and gets result.
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
       // val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        val imageHeight = mutable.height
        val imageWidth = mutable.width
        paint.textSize= imageHeight/15f
        paint.strokeWidth = imageHeight/85f

        scores.forEachIndexed { index, fl ->
            if(fl>0.5){
                val detectedClass = classes[index].toInt() //getting detected class as an integer
                paint.color = Color.argb(255, detectedClass * 50 % 255, detectedClass * 100 % 255, detectedClass * 150 % 255) // generating a unique color based on class
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(locations.get(index*4+1)*imageWidth, locations.get(index*4)*imageHeight, locations.get(index*4+3)*imageWidth, locations.get(index*4+2)*imageHeight),paint)
                paint.style = Paint.Style.FILL
                val label = labels[detectedClass] //mapping class to corresponding label

                // Add an offset for the text
                val textOffset = paint.textSize / 4
                canvas.drawText("$label ${fl*100}%", locations.get(index*4+1)*imageWidth, locations.get(index*4)*imageHeight - textOffset, paint)
            }
        }

        imageView.setImageBitmap(mutable)
    }

}