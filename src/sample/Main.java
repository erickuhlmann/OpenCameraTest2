package sample;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.beans.value.ChangeListener;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.jetbrains.annotations.NotNull;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.List;
import java.util.Vector;

public class Main extends Application {

    private static final boolean BLOWUP_EYE = false;
    private static final boolean DRAW_KELLY_MASKS = false;
    private static final boolean OUTLINE_FACES = false;
    private static final boolean OUTLINE_EYES = false;
    private static final boolean OUTLINE_MOUTHS = false;

    private static final Scalar MASK_COLOR = new Scalar(0,0,0,255); // black

    private static final int TIMER_INTERVAL = 250;

    @FXML
    private ImageView cameraView;

    @FXML
    private AnchorPane cameraPane;

    @FXML
    private BorderPane borderPane;

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private VideoCapture capture;
    private Timer timer;

    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;
    CascadeClassifier mouthCascade;

    @Override
    public void start(Stage primaryStage) throws Exception
    {
        System.out.println(Core.VERSION);

        FXMLLoader loader = new FXMLLoader();
        loader.setController(this);
        loader.setLocation(getClass().getResource("sample.fxml"));
        Parent root = loader.load();

        primaryStage.setTitle("Eyeblow");
        Scene primaryScene = new Scene(root, 640, 480);
        primaryStage.setScene(primaryScene);
        primaryStage.show();

        // get camera image to resize with window resize
        ChangeListener<Number> stageSizeListener = (observable, oldValue, newValue) ->
            resizeCameraView(primaryStage, primaryScene);
        primaryStage.widthProperty().addListener(stageSizeListener);
        primaryStage.heightProperty().addListener(stageSizeListener);
        resizeCameraView(primaryStage, primaryScene);


        // initialise classifiers
        faceCascade = new CascadeClassifier();
        faceCascade.load(makeFilePath("resources", "haarcascades", "haarcascade_frontalface_alt.xml"));
        eyeCascade = new CascadeClassifier();
        eyeCascade.load(makeFilePath("resources", "haarcascades", "haarcascade_eye.xml"));
        mouthCascade = new CascadeClassifier();
        mouthCascade.load(makeFilePath("resources", "haarcascades", "haarcascade_mcs_mouth.xml"));

        // initialise video capture
        capture = new VideoCapture();
        capture.open(0);

        // start draw timer
        // timer = new DrawTimer();
        // timer.start();

        ActionListener timerAction = new ActionListener()
        {
            private Mat frame = new Mat();

            @Override
            public void actionPerformed(ActionEvent e)
            {
                // capture a frame
                // Mat frame = new Mat();
                if (capture.read(frame))
                {
                    // process the frame
                    processFrame(frame);
                    // convert the frame into a javafx image
                    Image image = convertFrameToImage(frame);
                    // display the image
                    cameraView.setImage(image);
                }
            }
        };

        timer = new Timer(TIMER_INTERVAL, timerAction);
        timer.start();
    }

    @Override
    public void stop(){
        System.out.println("Stage is closing");
        timer.stop();
        // Save file
    }

    /**
     * Executes detection and painting results
     * @param frame
     */
    private void processFrame(Mat frame)
    {
        // prepare for face detection
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // do face detection
        int minFaceSize = Math.round(grayFrame.rows() * 0.1f);
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2,
                Objdetect.CASCADE_SCALE_IMAGE, new Size(minFaceSize,minFaceSize), new Size());

        // do eye detection
        int minEyeSize = Math.round(grayFrame.rows() * 0.05f);
        MatOfRect eyes = new MatOfRect();
        eyeCascade.detectMultiScale(grayFrame, eyes, 1.1, 2,
                Objdetect.CASCADE_SCALE_IMAGE, new Size(minEyeSize,minEyeSize), new Size());

        // do mouth detection
        int minMouthSize = Math.round(grayFrame.rows() * 0.2f);
        MatOfRect mouths = new MatOfRect();
        mouthCascade.detectMultiScale(grayFrame, mouths, 1.1, 2,
                Objdetect.CASCADE_SCALE_IMAGE, new Size(minMouthSize,minMouthSize), new Size());

        // get Rect arrays from detected rectangles
        Rect[] facesArray = faces.toArray();
        Rect[] eyesArray = eyes.toArray();
        Rect[] mouthsArray = mouths.toArray();

        // draw blown up eye
        if (BLOWUP_EYE)
        {
            Rect er = blowupEyeRect(facesArray, eyesArray);
            if (er != null)
            {
                Mat sourceEyeMat = frame.submat(er);
                Mat blowupEyeMat = new Mat();
                Size blowupSize = new Size();
                blowupSize.width = frame.width();
                blowupSize.height = frame.height();
                Imgproc.resize(sourceEyeMat, blowupEyeMat, blowupSize);
                blowupEyeMat.copyTo(frame);
            }
        }

        // draw kelly masks
        if (DRAW_KELLY_MASKS)
        {
            drawKellyMasks(frame, facesArray, eyesArray);
        }

        // draw results of face detection to the original camera frame
        if (OUTLINE_FACES)
        {
            for (Rect rect : facesArray)
            {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0, 255), 3);
            }
        }

        // draw results of eye detection to the original camera frame
        if (OUTLINE_EYES)
        {
            for (Rect rect : eyesArray)
            {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(255, 255, 0, 255), 2);
            }
        }

        // draw results of mouth detection
        if (OUTLINE_MOUTHS)
        {
            for (Rect rect : mouthsArray)
            {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 0, 255, 255), 2);
            }
        }
    }

    /**
     * Returns the rect of the first detected eye in the first detected face.
     * If there was no detected face then returns rect of first detected eye.
     * Returns null if there was no detected eye.
     * @param faces
     * @param eyes
     * @return
     */
    private Rect blowupEyeRect(@NotNull Rect @NotNull [] faces, @NotNull Rect[] eyes)
    {
        Rect eyeRect = null;

        for (Rect f : faces)
        {
            for (Rect e : eyes)
            {
                if (rect1InRect2(e,f))
                {
                    eyeRect = e;
                    break;
                }
            }
            if (eyeRect != null)
                break;
        }

        if (eyeRect == null)
        {
            if (eyes.length > 0)
            {
                eyeRect = eyes[0];
            }
        }

        return eyeRect;
    }

    private Image convertFrameToImage(Mat frame)
    {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }

    private void drawKellyMasks(@NotNull Mat frame, @NotNull Rect[] faces, @NotNull Rect[] eyes)
    {
        // draw a mask for each face
        for (Rect face: faces)
        {
            // get all the eyes in this face
            List<Rect> faceEyes = new Vector<Rect>();
            for (Rect eye: eyes)
            {
                if (rect1InRect2(eye, face))
                {
                    faceEyes.add(eye);
                }
            }

            // if there are eyes in this face then draw the mask with eyes slit
            if (faceEyes.size() > 0)
            {
                // compute the bounding rectangle for all eyes in this face
                double left = minLeft(faceEyes);
                double top = minTop(faceEyes);
                double right = maxRight(faceEyes);
                double bottom = maxBottom(faceEyes);
                Rect eyesRect = new Rect(new Point(left,top), new Point(right, bottom));

                // draw the mask
                Point pA = face.tl();
                Point pB = new Point(eyesRect.tl().x, face.br().y);
                Point pC = new Point(eyesRect.tl().x, face.tl().y);
                Point pD = new Point(eyesRect.br().x, eyesRect.tl().y);
                Point pE = new Point(eyesRect.tl().x, eyesRect.br().y);
                Point pF = new Point(eyesRect.br().x, face.br().y);
                Point pG = new Point(eyesRect.br().x, face.tl().y);
                Point pH = face.br();
                Imgproc.rectangle(frame, pA, pB, MASK_COLOR, Imgproc.FILLED);
                Imgproc.rectangle(frame, pC, pD, MASK_COLOR, Imgproc.FILLED);
                Imgproc.rectangle(frame, pE, pF, MASK_COLOR, Imgproc.FILLED);
                Imgproc.rectangle(frame, pG, pH, MASK_COLOR, Imgproc.FILLED);
            }
            // otherwise, draw a face with no eyes
            else
            {
                //Imgproc.rectangle(frame, face.tl(), face.br(), MASK_COLOR, Imgproc.FILLED);
            }
        }
    }

    private static double minLeft(List<Rect> rects)
    {
        double result = rects.get(0).tl().x;
        for (int i = 1; i < rects.size(); i++)
        {
            if (rects.get(i).tl().x < result)
            {
                result = rects.get(i).tl().x;
            }
        }
        return result;
    }

    private static double minTop(List<Rect> rects)
    {
        double result = rects.get(0).tl().y;
        for (int i = 1; i < rects.size(); i++)
        {
            if (rects.get(i).tl().y < result)
            {
                result = rects.get(i).tl().y;
            }
        }
        return result;
    }

    private static double maxRight(List<Rect> rects)
    {
        double result = rects.get(0).br().x;
        for (int i = 1; i < rects.size(); i++)
        {
            if (rects.get(i).br().x > result)
            {
                result = rects.get(i).br().x;
            }
        }
        return result;
    }

    private static double maxBottom(List<Rect> rects)
    {
        double result = rects.get(0).br().y;
        for (int i = 1; i < rects.size(); i++)
        {
            if (rects.get(i).br().y > result)
            {
                result = rects.get(i).br().y;
            }
        }
        return result;
    }

    private static boolean rect1InRect2(Rect rect1, Rect rect2)
    {
        return (rect1.tl().x >= rect2.tl().x) && (rect1.tl().y >= rect2.tl().y)
                && (rect1.br().x <= rect2.br().x) && (rect1.br().y <= rect2.br().y);
    }

    public static void main(String[] args)
    {
        launch(args);
    }

    /**
     * Resizes the CameraView to fill the stage (window)
     * @param stage
     * @param scene
     */
    private void resizeCameraView(Stage stage, Scene scene)
    {
        cameraView.setFitWidth(stage.getWidth());
        cameraView.setFitHeight(stage.getHeight()-20);
    }

/*
    private class DrawTimer extends AnimationTimer
    {
        @Override
        public void handle(long now)
        {
            // capture a frame
            Mat frame = new Mat();
            if (capture.read(frame))
            {
                // process the frame
                //processFrame(frame);

                // convert the frame into a javafx image
                Image image = convertFrameToImage(frame);

                // display the image
                cameraView.setImage(image);
            }
        }
    }
*/

    private String makeFilePath(String ...names)
    {
        String result = "";
        if (names.length > 0)
        {
            result = names[0];
            for (int i = 1; i < names.length; i++)
            {
                result = result + File.separator + names[i];
            }
        }
        System.out.println(result);
        return result;
    }
}
