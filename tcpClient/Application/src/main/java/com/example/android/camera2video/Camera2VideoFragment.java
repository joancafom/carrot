/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camera2video;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.media.Image;
import android.media.ImageReader;
import android.support.v4.app.DialogFragment;
import android.support.v4.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/*
    "Fragment" -> Static library support version of the framework's Fragment.
        A Fragment is a piece of an application's user interface or behavior that can be placed in
        an Activity.
    "View.OnClickListener" -> Interface definition for a callback to be invoked when a view is
        clicked.
 */

public class Camera2VideoFragment extends Fragment
        implements View.OnClickListener {

    // ---------------------------------------- CONSTANTS ------------------------------------------

    /*
        "SparseIntArrays" map integers to integers. Unlike a normal array of integers, there can be
        gaps in the indices. It is intended to be more memory efficient than using a HashMap to map
        Integers to Integers.
     */

    private static final int SENSOR_ORIENTATION_DEFAULT_DEGREES = 90;
    private static final int SENSOR_ORIENTATION_INVERSE_DEGREES = 270;
    private static final SparseIntArray DEFAULT_ORIENTATIONS = new SparseIntArray();
    private static final SparseIntArray INVERSE_ORIENTATIONS = new SparseIntArray();

    private static final String TAG = "Camera2VideoFragment";
    private static final int REQUEST_VIDEO_PERMISSIONS = 1;
    private static final String FRAGMENT_DIALOG = "dialog";

    /*
        "CAMERA" - Required to be able to access the camera device.
        "RECORD_AUDIO " - Allows an application to record audio.
         TODO: Eliminar la grabación de audio, que no es útil en nuestro contexto.
     */

    private static final String[] VIDEO_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
    };

    /*
        A Surface is generally created by or from a consumer of image buffers (SurfaceTexture,
        MediaRecorder, Allocation...) and is handed to some kind of producer (OpenGL, MediaPlayer,
        CameraDevice...) to draw into.

        ROTATION_0 - Rotation constant: 0 degree rotation (natural orientation).
            Constant value: 0 (0x00000000)
        ROTATION_90 - Rotation constant: 90 degree rotation.
            Constant value: 1 (0x00000001)
        ROTATION_180 - Rotation constant: 180 degree rotation.
            Constant value: 2 (0x00000002)
        ROTATION_270 - Rotation constant: 270 degree rotation.
            Constant value: 3 (0x00000003)
     */

    static {
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_0, 90);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_90, 0);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_180, 270);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    static {
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_0, 270);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_90, 180);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_180, 90);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_270, 0);
    }

    // An AutoFitTextureView for the camera preview.
    private AutoFitTextureView mTextureView;

    // Button to record the video
    private Button mButtonVideo;

    /*
        A reference to the opened CameraDevice.

        The "CameraDevice" class is a representation of a single camera connected to an Android
        device, allowing for fine-grain control of image capture and post-processing at high frame
        rates.
     */
    private CameraDevice mCameraDevice;

    /*
        A reference to the current CameraCaptureSession.

        The "CameraCaptureSession" is a configured capture session for a CameraDevice, used for
        capturing images from the camera or reprocessing images captured from the camera in the same
        session previously.
     */
    private CameraCaptureSession mPreviewSession;

    /*
        The SurfaceTextureListener handles several lifecycle events on a TextureView.

        This listener can be used to be notified when the surface texture associated with this
        texture view is available.

        A "TextureView" can be used to display a content stream. Such a content stream can for
        instance be a video or an OpenGL scene. The content stream can come from the application's
        process as well as a remote process.
     */
    private TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {

        /*
            Invoked when a TextureView's SurfaceTexture is ready for use.

            @param surfaceTexture - The surface returned by TextureView. getSurfaceTexture()
            @param width - The width of the surface.
            @param height - The height of the surface.
         */
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture,
                                              int width, int height) {
            openCamera(width, height);
        }

        /*
            Invoked when the specified SurfaceTexture is about to be destroyed. If returns true, no
            rendering should happen inside the surface texture after this method is invoked.

            @param surfaceTexture - The surface about to be destroyed.
         */
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }

        /*
            Invoked when the SurfaceTexture's buffers size changed.

            @param surfaceTexture - The surface returned by TextureView. getSurfaceTexture()
            @param width - The new width of the surface.
            @param height - The new height of the surface.
         */
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture,
                                                int width, int height) {
            configureTransform(width, height);
        }

        /*
            Invoked when the specified SurfaceTexture is updated through
            SurfaceTexture.updateTexImage().

            @param surfaceTexture - The surface just updated.
         */
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        }

    };

    // "Size" is an immutable class for describing width and height dimensions in pixels.

    // The Size of the camera preview.
    private Size mPreviewSize;
    // The Size of the video recording.
    private Size mVideoSize;

    // The "MediaRecorder" is used to record audio and video.
    // TODO: Eliminar la grabación de audio, que no es útil en nuestro contexto.
    private MediaRecorder mMediaRecorder;
    // Whether the app is recording video now
    private boolean mIsRecordingVideo;

    /*
        "mBackgroundThread" is an additional thread for running tasks that shouldn't block the UI.

        "HandlerThread" is a handy class for starting a new thread that has a looper. The looper can
        then be used to create handler classes. Note that start() must still be called.
     */
    private HandlerThread mBackgroundThread;

    /*
        "mBackgorundHandler" is a Handler for running tasks in the background.

        A "Handler" allows you to send and process Message and Runnable objects associated with a
        thread's MessageQueue. Each Handler instance is associated with a single thread an that
        thread's message queue.
     */
    private Handler mBackgroundHandler;

    /*
        "mCameraOpenCloseLock" is a Semaphore to prevent the app from exiting before closing the
        camera.

        A "Semaphore" is a counting semaphore. Conceptually, a semaphore maintains a set of permits.
        Each acquire() blocks if necessary until a permit is available, and then takes it. Each
        release() adds a permit, potentially releasing a blocking acquirer. However, no actual
        permit objects are used; the Semaphore just keeps a count of the number available and acts
        accordingly.

        @constructor Semaphore(int permits) - Creates a Semaphore with the given number of permits
        and nonfair fairness setting.
     */
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /*
        "CameraDevice.StateCallback" is called when "CameraDevice" changes its status.

        "CameraDevice.StateCallback" is a callback objects for receiving updates about the state of
        a camera device.
     */
    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {


        /*
            This method is called when a camera device is no longer available for use.

            @param cameraDevice - The device that has been disconnected.
         */
        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            /*
                Semaphore public void release()

                Releases a permit, returning it to the semaphore.
                Releases a permit, increasing the number of available permits by one. If any threads
                are trying to acquire a permit, then one is selected and given the permit that was
                just released. That thread is (re)enabled for thread scheduling purposes.
             */
            mCameraOpenCloseLock.release();
            /*
                CameraDevice public abstract void close()

                Close the connection to this camera device as quickly as possible.
             */
            cameraDevice.close();
            mCameraDevice = null;
        }

        /*
            This method is called when a camera device has encountered a serious error.

            @param cameraDevice - The device reporting the error.
            @param error - The error code.
         */
        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            /*
                Semaphore public void release()

                Releases a permit, returning it to the semaphore.
                Releases a permit, increasing the number of available permits by one. If any threads
                are trying to acquire a permit, then one is selected and given the permit that was
                just released. That thread is (re)enabled for thread scheduling purposes.
             */
            mCameraOpenCloseLock.release();
            /*
                CameraDevice public abstract void close()

                Close the connection to this camera device as quickly as possible.
             */
            cameraDevice.close();
            mCameraDevice = null;

            /*
                public final Activity getActivity()
                - Return the Activity this fragment is currently associated with.
             */
            Activity activity = getActivity();
            if (null != activity) {
                /*
                    public void finish()
                    - Call this when your activity is done and should be closed. The ActivityResult
                    is propagated back to whoever launched you via onActivityResult().
                 */
                activity.finish();
            }
        }

        /*
            This method is called when a camera device has finished opening.

            @param cameraDevice - The camera device that has become opened.
         */
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
            /*
                Semaphore public void release()

                Releases a permit, returning it to the semaphore.
                Releases a permit, increasing the number of available permits by one. If any threads
                are trying to acquire a permit, then one is selected and given the permit that was
                just released. That thread is (re)enabled for thread scheduling purposes.
             */
            mCameraOpenCloseLock.release();

            if (null != mTextureView) {
                configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
            }
        }
    };

    private Integer mSensorOrientation;
    private String mNextVideoAbsolutePath;

    /*
        A "CaptureRequest.Builder" is a builder for capture requests.

        A "CaptureRequest" is an immutable package of settings and outputs needed to capture a
        single image from the camera device.
        Contains the configuration for the capture hardware (sensor, lens, flash), the processing
        pipeline, the control algorithms, and the output buffers. Also contains the list of target
        Surfaces to send image data to for this capture.
     */
    private CaptureRequest.Builder mPreviewBuilder;
    private ImageReader mImageReader;

    // ---------------------------------------------------------------------------------------------

    private Socket socket;
    private static final int SERVERPORT = 447;
    //private static final String SERVER_IP = "192.168.43.78";
    private static final String SERVER_IP = "192.168.0.18";

    int[] mRgbBuffer;

    public static Camera2VideoFragment newInstance() {
        return new Camera2VideoFragment();
    }

    /*
        In this sample, we choose a video size with 3x4 aspect ratio. Also, we don't use sizes
        larger than 1080p, since MediaRecorder cannot handle such a high-resolution video.

        @param choices - The list of available sizes.
        @return The video size.

        TODO: Cambiar el aspect ratio que vamos a utilizar.
        TODO: Cambiar la resolución que vamos a utilizar.
     */
    private static Size chooseVideoSize(Size[] choices) {
        for (Size size : choices) {
            // Comprueba si los tamaños cumplen el ratio y resolución
            if (size.getWidth() == size.getHeight() * 4 / 3 && size.getWidth() <= 1080) {
                return size;
            }
        }
        Log.e(TAG, "No se ha encontrado ningún tamaño de vídeo adecuado a los requisitos de" +
                "ratio y resolución.");
        return choices[choices.length - 1];
    }

    /*
        Given "choices" of "Size"'s supported by a camera, chooses the smallest one whose width and
        height are at least as large as the respective requested values, and whose aspect ratio
        matches with the specified value.

        @param choices - The list of sizes that the camera supports for the intended output.
        @param width - The minimum desired width.
        @param height - The minimum desired height.
        @param aspectRatio - The aspect ratio.
        @return - The optimal "Size", or an arbitrary one if none were big enough.
     */
    private static Size chooseOptimalSize(Size[] choices, int width, int height, Size aspectRatio) {
        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getHeight() == option.getWidth() * h / w &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }

        // Pick the smallest of those, assuming we found any
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else {
            Log.e(TAG, "No se ha encontrado ningún tamaño de previsualización adecuado.");
            return choices[0];
        }
    }

    /*
        (View) onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState)
        - Called to have the fragment instantiate its user interfance view.

        @param inflater - The LayoutInflater object that can be used to inflate any views in the
                            fragment.
        @param container - This is the parent view that the fragment's UI should be attached to. The
                            fragment should not add the view itself, but this can be used to
                            generate the LayoutParams of the view.
        @param - savedInstanceState - This fragment is being re-constructed from a previous saved
                                        state as given here.
        @return - Return the View for the fragment's UI, or null.
     */
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        /*
            (View) inflate(int resource, ViewGroup root, boolean attachToRoot)
            - Inflate a new view hierarchy from the specified xml resource.

            @param resource - ID from an XML layout resource to load.
            @param root - Optional view to be the parent of the generated hierarchy (if attachToRoot
                            is true), or else simply an object that provides a set of LayoutParams
                            values for root of the returned hierarchy (if attachToRoot is false).
            @param attachToRoot - Whether the inflated hierarchy should be attached to the root
                                    parameter. If false, root is only used to create the correct
                                    subclass of LayoutParams for the root view in the XML.
            @return - The root view of the inflated hierarchy. If root was supplied and attachToRoot
                        is true, this is root; otherwise it is the root of the inflated XML file.
         */
        return inflater.inflate(R.layout.fragment_camera2_video, container, false);
    }

    /*
        (Fragment) void onViewCreated(View view, Bundle savedInstanceState)
        - Called immediately after onCreateView(LayoutInflater, ViewGroup, Bundle) has returned, but
        before any saved state has been restored in to the view. This gives subclasses a chance to
        initialize themselves once they know their view hierarchy has been completely created. The
        fragment's view hierarchy is not however attached to its parent at this point.

        @param view - The View returned by onCreateView(LayoutInflater, ViewGroup, Bundle)
        @param savedInstanceState - If non-null, this fragment is being re-constructed from a
                                    previous saved state as given here.
     */
    @Override
    public void onViewCreated(@NonNull final View view, Bundle savedInstanceState) {
        /*
            (View) public final T findViewById(int id)
            - Finds the first descendant view with the given ID, the view itself if the ID matches
            getId(), or null if the ID is invalid (<0) or there is no matching view in the hierarchy.
         */
        mTextureView = view.findViewById(R.id.texture);
        mButtonVideo = view.findViewById(R.id.video);

        /*
            (View) public void setOnClickListener(View.OnClickListener l)
            - Register a callback to be invoked when this view is clicked. If the view is not
            clickable, it becomes clickable.

            @param l - The callback that will run.
         */
        mButtonVideo.setOnClickListener(this);
        view.findViewById(R.id.info).setOnClickListener(this);
    }

    /*
      (Fragment) void onResume()
      - Called when the fragment is visible to the user and actively running. This is generally tied
      to Activity.onResume of the containing Activity's lifecycle.
     */
    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
        /*
            (TextureView) public boolean isAvailable()
            - Returns true if the SurfaceTexture associated with this TextureView is available for
            rendering. When this method returns true, getSurfaceTexture() returns a valid surface
            texture.
         */
        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            /*
                (TextureView) public void
                setSurfaceTextureListener(TextureView.SurfaceTextureListener listener)
                - Sets the TextureView.SurfaceTextureListener used to listen to surface texture
                events.

                @param listener
             */
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    /*
        (Fragment) void onPause()
        - Called when the Fragment is no longer resumed. This is generally tied to Activity.onPause
        of the containing Activity's lifecycle.
     */
    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    /*
        (View.OnClickListener) public abstract void onClick(View v)
        - Called when a view has been clicked.

        @param v - The view that was clicked.
     */
    @Override
    public void onClick(View view) {
        // Comprobar en que Listener se ha hecho click.
        switch (view.getId()) {
            // Se ha hecho click en el botón de vídeo
            case R.id.video: {
                // Si se está grabando
                if (mIsRecordingVideo) {
                    // Parar la grabación
                    stopRecordingVideo();
                // Si no se está grabando
                } else {
                    // Comenzar la grabación
                    startRecordingVideo();
                }
                break;
            }
            // Se ha hecho click en el botón de info
            case R.id.info: {
                Activity activity = getActivity();
                if (null != activity) {
                    // Se crea un mensaje de diálogo (AlertDialog.Builder(Context context))
                    new AlertDialog.Builder(activity)
                            .setMessage(R.string.intro_message)
                            .setPositiveButton(android.R.string.ok, null)
                            .show();
                }
                break;
            }
        }
    }

    /*
        private void startBackgroundThread()
        - Starts a background thread and its Handler.

        A Thread is a thread of execution in a program. The Java Virtual Machine allows an
        application to have multiple threads of execution running concurrently.

        A Handler allows you to send and process Message and Runnable objects associated with a
        thread's MessageQueue. Each Handler instance is associated with a single thread and that
        thread's message queue. When you create a new Handler, it is bound to the thread / message
        queue of the thread that is creating it - from that point on, it will deliver messages and
        runnables to that message queue and execute them as they come out of the message queue.
     */
    private void startBackgroundThread() {
        /*
            HandlerThread is a handy class for starting a new thread that has a looper. The looper
            can then be used to create handler classes. Note that start() must still be called.
         */
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /*
        private void stopBackgroundThread()
        - Stops the background thread and its Handler.
     */
    private void stopBackgroundThread() {
        /*
            (HandlerThread) public boolean quitSafely()
            - Quits the handler thread's looper safely. Causes the handler thread's looper to
            terminate as soon as all remaining messages in the message queue that are already due to
            be delivered have been handled. Pending delayed messages with due times in the future
            will not be delivered.
         */
        mBackgroundThread.quitSafely();
        try {
            /*
                (Handler) public final void join()
                - Waits for this thread to die. Throws an InterruptedException if any thread has
                interrupted the current thread. The interrupted status of the current thread is
                cleared when this exception is thrown.
             */
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /*
        private boolean shouldShowRequestPermissionRationale(String[] permissions)
        - Gets whether you should show UI with rationale for requesting permissions.

        @param permissions - The permissions your app wants to request.
        @return - Whether you can show permission rationale UI.
     */
    private boolean shouldShowRequestPermissionRationale(String[] permissions) {
        for (String permission : permissions) {
            /*
                (Fragment) public boolean shouldShowRequestPermissionRationale(String permission)
                - Gets whether you should show UI with rationale for requesting a permission. You
                should do this only if you do no t have the permission and the context in which the
                permission is requested does not clearly communicate to the user what would be the
                benefit from granting this permission.
             */
            if (shouldShowRequestPermissionRationale(permission)) {
                return true;
            }
        }
        return false;
    }

    /*
        private void requestVideoPermissions()
        - Requests permissions needed for recording video.
     */
    private void requestVideoPermissions() {
        // Si se debe mostrar en la UI
        if (shouldShowRequestPermissionRationale(VIDEO_PERMISSIONS)) {
            // Se muestra un diálogo
            new ConfirmationDialog().show(getChildFragmentManager(), FRAGMENT_DIALOG);
        // Si el usuario ya ha concedido los permisos
        } else {
            /*
                (Fragment) public final void requestPermissions(String[] permissions, int requestCode)
                - Request permissions to be granted to this application. These permissions must be
                requested in your manifest, they should not be granted to your app, and they should
                have protection level #PROTECTION_DANGEROUS dangerous, regardless whether they are
                declared by the platform or a third-party app.

                @param permissions - The requested permissions.
                @param requestCode - Application specific request code to match with a result
                                        reported to onRequestPermissionsResult(int, String[], int[]).
             */
            requestPermissions(VIDEO_PERMISSIONS, REQUEST_VIDEO_PERMISSIONS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        Log.d(TAG, "onRequestPermissionsResult");
        if (requestCode == REQUEST_VIDEO_PERMISSIONS) {
            if (grantResults.length == VIDEO_PERMISSIONS.length) {
                for (int result : grantResults) {
                    if (result != PackageManager.PERMISSION_GRANTED) {
                        ErrorDialog.newInstance(getString(R.string.permission_request))
                                .show(getChildFragmentManager(), FRAGMENT_DIALOG);
                        break;
                    }
                }
            } else {
                ErrorDialog.newInstance(getString(R.string.permission_request))
                        .show(getChildFragmentManager(), FRAGMENT_DIALOG);
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private boolean hasPermissionsGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ActivityCompat.checkSelfPermission(getActivity(), permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void getRGBIntFromPlanes(Image.Plane[] planes) {
        ByteBuffer yPlane = planes[0].getBuffer();
        ByteBuffer uPlane = planes[1].getBuffer();
        ByteBuffer vPlane = planes[2].getBuffer();

        int bufferIndex = 0;
        final int total = yPlane.capacity();
        final int uvCapacity = uPlane.capacity();
        final int width = planes[0].getRowStride();

        int yPos = 0;
        for (int i = 0; i < mPreviewSize.getHeight(); i++) {
            int uvPos = (i >> 1) * width;

            for (int j = 0; j < width; j++) {
                if (uvPos >= uvCapacity - 1) {
                    break;
                }
                if (yPos >= total) {
                    break;
                }

                final int y1 = yPlane.get(yPos++) & 0xff;
                final int u = (uPlane.get(uvPos) & 0xff) - 128;
                final int v = (vPlane.get(uvPos + 1) & 0xff) - 128;

                if ((j & 1) == 1) {
                    uvPos += 2;
                }

                final int y1192 = 1192 * y1;
                int r = (y1192 + 1634 * v);
                int g = (y1192 - 833 * v - 400 * u);
                int b = (y1192 + 2066 * u);

                r = (r < 0) ? 0 : ((r > 262143) ? 262143 : r);
                g = (g < 0) ? 0 : ((g > 262143) ? 262143 : g);
                b = (b < 0) ? 0 : ((b > 262143) ? 262143 : b);

                mRgbBuffer[bufferIndex++] = ((r << 6) & 0xff0000) |
                        ((g >> 2) & 0xff00) |
                        ((b >> 10) & 0xff);
            }
        }
    }

    public static byte[] yuvImageToByteArray(Image image) {

        assert(image.getFormat() == ImageFormat.YUV_420_888);

        int width = image.getWidth();
        int height = image.getHeight();

        Image.Plane[] planes = image.getPlanes();
        byte[] result = new byte[width * height * 3 / 2];

        int stride = planes[0].getRowStride();
        if (stride == width) {
            planes[0].getBuffer().get(result, 0, width);
        }
        else {
            for (int row = 0; row < height; row++) {
                planes[0].getBuffer().position(row*stride);
                planes[0].getBuffer().get(result, row*width, width);
            }
        }

        stride = planes[1].getRowStride();
        assert (stride == planes[2].getRowStride());
        byte[] rowBytesCb = new byte[stride];
        byte[] rowBytesCr = new byte[stride];

        for (int row = 0; row < height/2; row++) {
            int rowOffset = width*height + width/2 * row;
            planes[1].getBuffer().position(row*stride);
            planes[1].getBuffer().get(rowBytesCb, 0, width/2);
            planes[2].getBuffer().position(row*stride);
            planes[2].getBuffer().get(rowBytesCr, 0, width/2);

            for (int col = 0; col < width/2; col++) {
                result[rowOffset + col*2] = rowBytesCr[col];
                result[rowOffset + col*2 + 1] = rowBytesCb[col];
            }
        }
        return result;
    }

    private static byte[] NV21toJPEG(byte[] nv21, int width, int height, int quality) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        yuv.compressToJpeg(new Rect(0, 0, width, height), quality, out);
        return out.toByteArray();
    }

    private static byte[] YUV420toNV21(Image image) {
        Rect crop = image.getCropRect();
        int format = image.getFormat();
        int width = crop.width();
        int height = crop.height();
        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[width * height * ImageFormat.getBitsPerPixel(format) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        int channelOffset = 0;
        int outputStride = 1;
        for (int i = 0; i < planes.length; i++) {
            switch (i) {
                case 0:
                    channelOffset = 0;
                    outputStride = 1;
                    break;
                case 1:
                    channelOffset = width * height + 1;
                    outputStride = 2;
                    break;
                case 2:
                    channelOffset = width * height;
                    outputStride = 2;
                    break;
            }

            ByteBuffer buffer = planes[i].getBuffer();
            int rowStride = planes[i].getRowStride();
            int pixelStride = planes[i].getPixelStride();

            int shift = (i == 0) ? 0 : 1;
            int w = width >> shift;
            int h = height >> shift;
            buffer.position(rowStride * (crop.top >> shift) + pixelStride * (crop.left >> shift));
            for (int row = 0; row < h; row++) {
                int length;
                if (pixelStride == 1 && outputStride == 1) {
                    length = w;
                    buffer.get(data, channelOffset, length);
                    channelOffset += length;
                } else {
                    length = (w - 1) * pixelStride + 1;
                    buffer.get(rowData, 0, length);
                    for (int col = 0; col < w; col++) {
                        data[channelOffset] = rowData[col * pixelStride];
                        channelOffset += outputStride;
                    }
                }
                if (row < h - 1) {
                    buffer.position(buffer.position() + rowStride - length);
                }
            }
        }
        return data;
    }

    ImageReader.OnImageAvailableListener mImageAvailable = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            Image image = reader.acquireLatestImage();
            if (image == null) {
                return;
            }

            //final Image.Plane[] planes = image.getPlanes();
            //final int total = planes[0].getRowStride() * mPreviewSize.getHeight();
            //if (mRgbBuffer == null || mRgbBuffer.length < total) {
            //    mRgbBuffer = new int[total];
            //}

            //getRGBIntFromPlanes(planes);

            byte[] data = NV21toJPEG(YUV420toNV21(image), image.getWidth(), image.getHeight(), 100);

            // byte[] imgbyte = yuvImageToByteArray(image);
            //Bitmap bmp=BitmapFactory.decodeByteArray(imgbyte,0,imgbyte.length);
            //saveToInternalStorage(bmp);
            String imgString = Base64.encodeToString(data, Base64.NO_WRAP);



            try {

                PrintWriter out = new PrintWriter(new BufferedWriter(
                        new OutputStreamWriter(socket.getOutputStream())),
                        true);
                //OutputStream output = socket.getOutputStream();
                //output.write(imgbyte);
                //out.println("-------------------------------------------");
                out.print(imgString);
                out.print("****");
                //output.flush();
                //out.write(imgbyte);
                out.flush();

                // Imagen a byte []
                // OutputStream output = socket.getOutputStream();
                // output.write(imgbyte);
                // output.flush();

            } catch (Exception e) {

            }

            image.close();
        }
    };

    private String saveToInternalStorage(Bitmap bitmapImage){
        ContextWrapper cw = new ContextWrapper(getActivity());
        // path to /data/data/yourapp/app_data/imageDir
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir
        File mypath=new File(directory,"profile.jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            // Use the compress method on the BitMap object to write image to the OutputStream
            bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }

    /**
     * Tries to open a {@link CameraDevice}. The result is listened by `mStateCallback`.
     */
    @SuppressWarnings("MissingPermission")
    private void openCamera(int width, int height) {
        new Thread(new ClientThread()).start();
        if (!hasPermissionsGranted(VIDEO_PERMISSIONS)) {
            requestVideoPermissions();
            return;
        }
        final Activity activity = getActivity();
        if (null == activity || activity.isFinishing()) {
            return;
        }
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            Log.d(TAG, "tryAcquire");
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            String cameraId = manager.getCameraIdList()[0];

            // Choose the sizes for camera preview and video recording
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics
                    .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            mSensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (map == null) {
                throw new RuntimeException("Cannot get available preview/video sizes");
            }
            mVideoSize = chooseVideoSize(map.getOutputSizes(MediaRecorder.class));
            mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                    width, height, mVideoSize);


            mImageReader = ImageReader.newInstance(mVideoSize.getWidth(), mVideoSize.getHeight(), ImageFormat.YUV_420_888, 3);
            mImageReader.setOnImageAvailableListener(mImageAvailable, mBackgroundHandler);




            int orientation = getResources().getConfiguration().orientation;
            if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                mTextureView.setAspectRatio(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            } else {
                mTextureView.setAspectRatio(mPreviewSize.getHeight(), mPreviewSize.getWidth());
            }
            configureTransform(width, height);
            mMediaRecorder = new MediaRecorder();
            manager.openCamera(cameraId, mStateCallback, null);
        } catch (CameraAccessException e) {
            Toast.makeText(activity, "Cannot access the camera.", Toast.LENGTH_SHORT).show();
            activity.finish();
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            ErrorDialog.newInstance(getString(R.string.camera_error))
                    .show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.");
        }
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            closePreviewSession();
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mMediaRecorder) {
                mMediaRecorder.release();
                mMediaRecorder = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.");
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    /**
     * Start the camera preview.
     */
    private void startPreview() {
        if (null == mCameraDevice || !mTextureView.isAvailable() || null == mPreviewSize) {
            return;
        }
        try {
            closePreviewSession();
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            List surfaces = new ArrayList<>();

            Surface previewSurface = new Surface(texture);
            surfaces.add(previewSurface);
            mPreviewBuilder.addTarget(previewSurface);

            Surface readerSurface = mImageReader.getSurface();
            surfaces.add(readerSurface);
            mPreviewBuilder.addTarget(readerSurface);


            mCameraDevice.createCaptureSession(surfaces,
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            mPreviewSession = session;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Activity activity = getActivity();
                            if (null != activity) {
                                Toast.makeText(activity, "Failed", Toast.LENGTH_SHORT).show();
                            }
                        }
                    }, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * Update the camera preview. {@link #startPreview()} needs to be called in advance.
     */
    private void updatePreview() {
        if (null == mCameraDevice) {
            return;
        }
        try {
            setUpCaptureRequestBuilder(mPreviewBuilder);
            HandlerThread thread = new HandlerThread("CameraPreview");
            thread.start();
            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setUpCaptureRequestBuilder(CaptureRequest.Builder builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
    }

    /**
     * Configures the necessary {@link android.graphics.Matrix} transformation to `mTextureView`.
     * This method should not to be called until the camera preview size is determined in
     * openCamera, or until the size of `mTextureView` is fixed.
     *
     * @param viewWidth  The width of `mTextureView`
     * @param viewHeight The height of `mTextureView`
     */
    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }

    private void setUpMediaRecorder() throws IOException {
        final Activity activity = getActivity();
        if (null == activity) {
            return;
        }
        mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        if (mNextVideoAbsolutePath == null || mNextVideoAbsolutePath.isEmpty()) {
            mNextVideoAbsolutePath = getVideoFilePath(getActivity());
        }
        mMediaRecorder.setOutputFile(mNextVideoAbsolutePath);
        mMediaRecorder.setVideoEncodingBitRate(10000000);
        mMediaRecorder.setVideoFrameRate(30);
        mMediaRecorder.setVideoSize(mVideoSize.getWidth(), mVideoSize.getHeight());
        mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        switch (mSensorOrientation) {
            case SENSOR_ORIENTATION_DEFAULT_DEGREES:
                mMediaRecorder.setOrientationHint(DEFAULT_ORIENTATIONS.get(rotation));
                break;
            case SENSOR_ORIENTATION_INVERSE_DEGREES:
                mMediaRecorder.setOrientationHint(INVERSE_ORIENTATIONS.get(rotation));
                break;
        }
        mMediaRecorder.prepare();
    }

    private String getVideoFilePath(Context context) {
        final File dir = context.getExternalFilesDir(null);
        return (dir == null ? "" : (dir.getAbsolutePath() + "/"))
                + System.currentTimeMillis() + ".mp4";
    }

    private void startRecordingVideo() {
        if (null == mCameraDevice || !mTextureView.isAvailable() || null == mPreviewSize) {
            return;
        }
        try {
            closePreviewSession();
            setUpMediaRecorder();
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            mPreviewBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            List<Surface> surfaces = new ArrayList<>();

            // Set up Surface for the camera preview
            Surface previewSurface = new Surface(texture);
            surfaces.add(previewSurface);
            mPreviewBuilder.addTarget(previewSurface);

            // Set up Surface for the MediaRecorder
            Surface recorderSurface = mMediaRecorder.getSurface();
            surfaces.add(recorderSurface);
            mPreviewBuilder.addTarget(recorderSurface);

            // Start a capture session
            // Once the session starts, we can update the UI and start recording
            mCameraDevice.createCaptureSession(surfaces, new CameraCaptureSession.StateCallback() {

                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    mPreviewSession = cameraCaptureSession;
                    updatePreview();
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // UI
                            mButtonVideo.setText(R.string.stop);
                            mIsRecordingVideo = true;

                            // Start recording
                            mMediaRecorder.start();
                        }
                    });
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Activity activity = getActivity();
                    if (null != activity) {
                        Toast.makeText(activity, "Failed", Toast.LENGTH_SHORT).show();
                    }
                }
            }, mBackgroundHandler);
        } catch (CameraAccessException | IOException e) {
            e.printStackTrace();
        }

    }

    private void closePreviewSession() {
        if (mPreviewSession != null) {
            mPreviewSession.close();
            mPreviewSession = null;
        }
    }

    private void stopRecordingVideo() {
        // UI
        mIsRecordingVideo = false;
        mButtonVideo.setText(R.string.record);
        // Stop recording
        mMediaRecorder.stop();
        mMediaRecorder.reset();

        Activity activity = getActivity();
        if (null != activity) {
            Toast.makeText(activity, "Video saved: " + mNextVideoAbsolutePath,
                    Toast.LENGTH_SHORT).show();
            Log.d(TAG, "Video saved: " + mNextVideoAbsolutePath);
        }
        mNextVideoAbsolutePath = null;
        startPreview();
    }

    /**
     * Compares two {@code Size}s based on their areas.
     */
    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }

    }

    public static class ErrorDialog extends DialogFragment {

        private static final String ARG_MESSAGE = "message";

        public static ErrorDialog newInstance(String message) {
            ErrorDialog dialog = new ErrorDialog();
            Bundle args = new Bundle();
            args.putString(ARG_MESSAGE, message);
            dialog.setArguments(args);
            return dialog;
        }

        @Override
        public @NonNull Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(getArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {
                            activity.finish();
                        }
                    })
                    .create();
        }

    }

    public static class ConfirmationDialog extends DialogFragment {

        @Override
        public @NonNull Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.permission_request)
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            requestPermissions(VIDEO_PERMISSIONS,
                                    REQUEST_VIDEO_PERMISSIONS);
                        }
                    })
                    .setNegativeButton(android.R.string.cancel,
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    parent.getActivity().finish();
                                }
                            })
                    .create();
        }

    }

    class ClientThread implements Runnable {

        @Override
        public void run() {

            try {
                InetAddress serverAddr = InetAddress.getByName(SERVER_IP);

                socket = new Socket(serverAddr, SERVERPORT);

            } catch (UnknownHostException e1) {
                e1.printStackTrace();
            } catch (IOException e1) {
                e1.printStackTrace();
            }

        }

    }
}
