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

import android.content.Context;
import android.util.AttributeSet;
import android.view.TextureView;

/*
    This class is meant to adjust a TextureView to a specified aspect ratio.

    "TextureView" -> A TextureView can be used to display a content stream. Such a content stream
        can for instance be a video or an OpenGL scene. The content stream can come from the
        application's process as well as a remote process.
 */
public class AutoFitTextureView extends TextureView {

    // ---------------------------------------- CONSTANTS ------------------------------------------

    /*
        mRatioWidth - Relative horizontal size
        mRatioHeight - Relative vertical size
     */

    private int mRatioWidth = 0;
    private int mRatioHeight = 0;

    // --------------------------------------- CONSTRUCTORS ----------------------------------------

    public AutoFitTextureView(Context context) {
        this(context, null);
    }

    public AutoFitTextureView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public AutoFitTextureView(Context context, AttributeSet attrs, int defStyle) {

        /*
            Llamada a un constructor público de TextureView.

                context -> The context to associate this view with.
                attrs -> The attributes of the XML tag that is inflating the view.
                defStyle -> An attribute in the current theme that contains a reference to a style
                    resource that supplies default values for the view. Can be 0 to not look for
                    defaults.
         */

        super(context, attrs, defStyle);
    }

    // ---------------------------------------------------------------------------------------------

    /*
        Sets the aspect ratio for this view. The size of the view will be measured based on the
        ratio calculated from the parameters. Note that the actual sizes od parameters don't matter,
        that is, calling setAspectRatio(2, 3) and setAspectRatio(4, 6) make the same result.

        @param width - Relative horizontal size
        @param height - Relative vertical size
     */
    public void setAspectRatio(int width, int height) {
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException("El tamaño no puede ser negativo.");
        }
        mRatioWidth = width;
        mRatioHeight = height;

        /*
            (View) public void requestLayout()
            - Call this when something has invalidated the layout of this view. This will force
            the redraw with the changes by scheduling a layout pass of the view tree.
         */

        requestLayout();
    }

    /*
        (View) protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec)
        - Measure the view and its content to determine the measured width and the measured height.
        This method is invoked by measure(int, int) and should be overridden by subclasses to
        provide accurate and efficient measurement of their contents.

        @param widthMeasureSpec - Horizontal space requirements as imposed by the parent.
        @param heightMeasureSpec - Vertical space requirements as imposed by the parent.
     */

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);

        /*
            A MeasureSpec encapsulates the layout requirements passed from parent to child. Each
            MeasureSpec represents a requirement for either the width or the height. A MeasureSpec
            is comprised of a size and a mode (Unspecified, Exactly, At_most).

            (View.MeasureSpec) public static int getSize(int measureSpec)
            - Extracts the size from te supplied measure specification.
            @param measureSpec - The measure specification to extract the size from
            @return The size in pixels defined in the supplied measure specification
         */

        int width = MeasureSpec.getSize(widthMeasureSpec);
        int height = MeasureSpec.getSize(heightMeasureSpec);

        /*
            (View) protected final void setMeasuredDimension(int measuredWidth, int measuredHeight)
            - This method must be called by onMeasure(int, int) to store the measured width and
            measured height.
         */

        // Si no se han especificado proporciones, se guardan los tamaños tal cual.
        if (0 == mRatioWidth || 0 == mRatioHeight) {
            setMeasuredDimension(width, height);
        } else {
            // Si al aplicarle la proporción a la altura, esta queda más pequeña que el ancho, le
            // aplicamos la proporción al ancho y, al contrario, si al aplicar la proporción a la
            // altura, esta queda más grande que el ancho, le aplicamos la proporción al ancho.
            if (width < height * mRatioWidth / mRatioHeight) {
                setMeasuredDimension(width, width * mRatioHeight / mRatioWidth);
            } else {
                setMeasuredDimension(height * mRatioWidth / mRatioHeight, height);
            }
        }
    }

}
