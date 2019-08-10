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

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

/*
    "AppCompatActivity" -> Clase base para Activities que usan las funcionalidades
        de la action bar de la librería de soporte.
 */

public class CameraActivity extends AppCompatActivity {

    /*
        El método "void onCreate(Bundle savedInstaceState)" realiza
        una inicialización de todos los fragmentos.

        Un Fragment representa un comportamiento o una parte de la
        interfaz de usuario en una Activity.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        /*
            El método "setContentView(int layoutResID)" indica el diseño
            que se debe asociar a la Actividad.

            "R.layout.activity_camera" indica que se debe buscar en
            res -> layout -> activity_camera.xml
         */

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        /*
            Si es la primera vez que lanzamos la Activiad, 'savedInstanceState' será null,
            ya que no hay un estado anterior guardado.
         */

        if (null == savedInstanceState) {

            /*
                FragmentManager getSupportFragmentManager()
                - Return the FragmentManager for interacting with fragments associated with
                this activity.

                FragmentTransaction beginTransaction()
                - Start a series of edit operations on the Fragments associated with this
                FragmentManager.

                FragmentTransaction replace(int containerViewId, Fragment fragment)
                - Replace an existing fragment that was added to a container. It's the
                same as calling remove(Fragment) for all currently added fragments and
                then add(int, Fragment).

                "R.id.container" -> Indica res -> layout -> activity_camera.xml -> android:id -> container

                int commit()
                - Schedules a commit of this transaction. The commit does not happen immediately; it
                will be scheduled as work on the main thread to be done the next time that thread is
                ready.
             */

            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.container, Camera2VideoFragment.newInstance())
                    .commit();
        }
    }
}
