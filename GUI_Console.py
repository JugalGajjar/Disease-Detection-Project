# Importing Libraries
from tkinter import *
import pickle
import numpy as np

# GUI Console Code
root = Tk()
root.title('Disease Detector')
# root.iconbitmap("/home/dlinano/Desktop/JUGAL GAJJAR/Disease Detection Project/Main_Logo.ico")

welcome = Label(root, text='WELCOME  TO  DISEASE  DETECTOR',
                font=('Arial', 14))
welcome.grid(row=0, column=1, columnspan=3, padx=10, pady=5)

blank = Label(root, text='\t')
blank.grid(row=1, column=2, padx=20, pady=5)

options = ['Diabetes Prediction', 'Heart Disease Prediction',
           'Breast Cancer Prediction', 'Parkinsons Disease Prediction',
           'Chronic Kidney Disease Prediction']
selected = StringVar()
selected.set(options[1])

blank = Label(root, text='')
blank.grid(row=2, column=1)

drop = OptionMenu(root, selected, *options)
drop.grid(row=2, column=2, padx=2, pady=5)


def diabetes():
    root = Tk()
    root.title('Diabetes Prediction')
    # root.iconbitmap("Main_Logo.ico")

    # Text Boxes
    pregnancies = Entry(root, width=30)
    pregnancies.grid(row=0, column=1, padx=10, pady=10)

    glucose = Entry(root, width=30)
    glucose.grid(row=1, column=1, padx=10, pady=10)

    bp = Entry(root, width=30)
    bp.grid(row=2, column=1, padx=10, pady=10)

    skinthick = Entry(root, width=30)
    skinthick.grid(row=3, column=1, padx=10, pady=10)

    insulin = Entry(root, width=30)
    insulin.grid(row=4, column=1, padx=10, pady=10)

    bmi = Entry(root, width=30)
    bmi.grid(row=5, column=1, padx=10, pady=10)

    diapedfunc = Entry(root, width=30)
    diapedfunc.grid(row=6, column=1, padx=10, pady=10)

    age = Entry(root, width=30)
    age.grid(row=7, column=1, padx=10, pady=10)

    # Labels with Text Boxes
    pregnancies_label = Label(root, text='Pregnancies')
    pregnancies_label.grid(row=0, column=0, padx=10, pady=10)

    glucose_label = Label(root, text='Glucose')
    glucose_label.grid(row=1, column=0, padx=10, pady=10)

    bp_label = Label(root, text='Blood Pressure')
    bp_label.grid(row=2, column=0, padx=10, pady=10)

    skinthick_label = Label(root, text='Skin Thickness')
    skinthick_label.grid(row=3, column=0, padx=10, pady=10)

    insulin_label = Label(root, text='Insulin')
    insulin_label.grid(row=4, column=0, padx=10, pady=10)

    bmi_label = Label(root, text='BMI')
    bmi_label.grid(row=5, column=0, padx=10, pady=10)

    diapedfunc_label = Label(root, text='Pedigree Func.')
    diapedfunc_label.grid(row=6, column=0, padx=10, pady=10)

    age_label = Label(root, text='Age')
    age_label.grid(row=7, column=0, padx=10, pady=10)

    blank = Label(root, text='\t')
    blank.grid(row=8, column=0, padx=20, pady=5)

    # Function to Invoke Trained Model and Predict Results
    def predict_dia():
        preg = int(pregnancies.get())
        bm = float(bmi.get())
        ag = int(age.get())
        diapedfun = float(diapedfunc.get())
        ins = int(insulin.get())
        st = int(skinthick.get())
        glu = int(glucose.get())
        bp_val = int(bp.get())

        # Clearing Text Boxes
        pregnancies.delete(0, END)
        bmi.delete(0, END)
        age.delete(0, END)
        diapedfunc.delete(0, END)
        insulin.delete(0, END)
        skinthick.delete(0, END)
        glucose.delete(0, END)
        bp.delete(0, END)

        # Loading Fitted Model & Standard Scaler
        file = open('./Models/Diabetes_model_pkl', 'rb')
        model = pickle.load(file)
        file.close()
        file = open('./Models/Diabetes_scaler_pkl', 'rb')
        scaler = pickle.load(file)
        file.close()

        input_data = (
            preg,
            glu,
            bp_val,
            st,
            ins,
            bm,
            diapedfun,
            ag,
            )

        # Processing Input Data to Make it Suitable for Model
        np_array = np.asarray(input_data)
        data_reshaped = np_array.reshape(1, -1)
        std_data = scaler.transform(data_reshaped)

        prediction = model.predict(std_data)

        # Displaying Result
        if prediction[0] == 0:
            message = Tk()
            message.title('Prediction Result')
            Label(message, text='The Person does NOT have Diabetes',
                  padx=10, pady=10).pack(padx=10, pady=10)
            message.mainloop()
        else:
            message = Tk()
            message.title('Prediction Result')
            Label(message, text='The Person has Diabetes', padx=10,
                  pady=10).pack(padx=10, pady=10)
            message.mainloop()

    submit_button = Button(root, text='Predict', command=predict_dia,
                           pady=10)
    submit_button.grid(
        row=9,
        column=0,
        columnspan=2,
        padx=10,
        pady=10,
        ipadx=78,
        )

    root.mainloop()


def heart():
    root = Tk()
    root.title('Heart Disease Prediction')
    # root.iconbitmap("Main_Logo.ico")

    # Text Boxes
    age = Entry(root, width=30)
    age.grid(row=0, column=1, padx=10, pady=10)

    sex = Entry(root, width=30)
    sex.grid(row=1, column=1, padx=10, pady=10)

    cpt = Entry(root, width=30)
    cpt.grid(row=2, column=1, padx=10, pady=10)

    rbp = Entry(root, width=30)
    rbp.grid(row=3, column=1, padx=10, pady=10)

    chol = Entry(root, width=30)
    chol.grid(row=4, column=1, padx=10, pady=10)

    bloodsug = Entry(root, width=30)
    bloodsug.grid(row=5, column=1, padx=10, pady=10)

    recgr = Entry(root, width=30)
    recgr.grid(row=6, column=1, padx=10, pady=10)

    maxhrtrate = Entry(root, width=30)
    maxhrtrate.grid(row=7, column=1, padx=10, pady=10)

    exerinang = Entry(root, width=30)
    exerinang.grid(row=8, column=1, padx=10, pady=10)

    oldpeak = Entry(root, width=30)
    oldpeak.grid(row=9, column=1, padx=10, pady=10)

    slope = Entry(root, width=30)
    slope.grid(row=10, column=1, padx=10, pady=10)

    novessels = Entry(root, width=30)
    novessels.grid(row=11, column=1, padx=10, pady=10)

    thal = Entry(root, width=30)
    thal.grid(row=12, column=1, padx=10, pady=10)

    # Labels with Text Boxes
    age_label = Label(root, text='Age')
    age_label.grid(row=0, column=0, padx=10, pady=10)

    sex_label = Label(root, text='Sex (0/1)')
    sex_label.grid(row=1, column=0, padx=10, pady=10)

    cpt_label = Label(root, text='Chest Pain Type (0/1/2/3)')
    cpt_label.grid(row=2, column=0, padx=10, pady=10)

    rbp_label = Label(root, text='Resting Blood Pres.')
    rbp_label.grid(row=3, column=0, padx=10, pady=10)

    chol_label = Label(root, text='Cholestrol (mg/dl)')
    chol_label.grid(row=4, column=0, padx=10, pady=10)

    bloodsug_label = Label(root, text='Fasting Blood Sugar (0/1)')
    bloodsug_label.grid(row=5, column=0, padx=10, pady=10)

    recgr_label = Label(root, text='Resting ECG Res. (0/1)')
    recgr_label.grid(row=6, column=0, padx=10, pady=10)

    maxhrtrate_label = Label(root, text='Max Heart Rate')
    maxhrtrate_label.grid(row=7, column=0, padx=10, pady=10)

    exerinang_label = Label(root, text='Exercise Ind. Angina (0/1)')
    exerinang_label.grid(row=8, column=0, padx=10, pady=10)

    oldpeak_label = Label(root, text='Old Peak')
    oldpeak_label.grid(row=9, column=0, padx=10, pady=10)

    slope_label = Label(root, text='Slope (0/1/2)')
    slope_label.grid(row=10, column=0, padx=10, pady=10)

    novessels_label = Label(root, text='No. of Vessels (0-4)')
    novessels_label.grid(row=11, column=0, padx=10, pady=10)

    thal_label = Label(root, text='Thal (0/1/2/3)')
    thal_label.grid(row=12, column=0, padx=10, pady=10)

    blank = Label(root, text='\t')
    blank.grid(row=13, column=0, padx=10, pady=5)
    
    # Function to Invoke Trained Model and Predict Results
    def predict_heart():
        ag = int(age.get())
        se = int(sex.get())
        cp = int(cpt.get())
        rp = int(rbp.get())
        ch = int(chol.get())
        bs = int(bloodsug.get())
        ecg = int(recgr.get())
        hrt = int(maxhrtrate.get())
        anig = int(exerinang.get())
        op = float(oldpeak.get())
        slo = int(slope.get())
        ves = int(novessels.get())
        tha = int(thal.get())

        # Clearing Text Boxes
        age.delete(0, END)
        sex.delete(0, END)
        cpt.delete(0, END)
        rbp.delete(0, END)
        chol.delete(0, END)
        bloodsug.delete(0, END)
        recgr.delete(0, END)
        maxhrtrate.delete(0, END)
        exerinang.delete(0, END)
        oldpeak.delete(0, END)
        slope.delete(0, END)
        novessels.delete(0, END)
        thal.delete(0, END)

        # Loading Fitted Model
        file = open('./Models/HeartDisease_model_pkl', 'rb')
        model = pickle.load(file)
        file.close()

        input_data = (
            ag,
            se,
            cp,
            rp,
            ch,
            bs,
            ecg,
            hrt,
            anig,
            op,
            slo,
            ves,
            tha,
            )

        # Processing Input Data to Make it Suitable for Model
        np_array = np.asarray(input_data)
        data_reshaped = np_array.reshape(1, -1)

        prediction = model.predict(data_reshaped)

        # Displaying Result
        if prediction[0] == 0:
            message = Tk()
            message.title('Prediction Result')
            Label(message, text='The Person does NOT have Heart Disease'
                  , padx=10, pady=10).pack(padx=10, pady=10)
            message.mainloop()
        else:
            message = Tk()
            message.title('Prediction Result')
            Label(message, text='The Person does NOT have Heart Disease'
                  , padx=10, pady=10).pack(padx=10, pady=10)
            message.mainloop()

    submit_button = Button(root, text='Predict', command=predict_heart,
                           pady=10)
    submit_button.grid(
        row=14,
        column=0,
        columnspan=2,
        padx=10,
        pady=10,
        ipadx=78,
        )

    root.mainloop()


def breast():
    root = Tk()
    root.title('Breast Cancer Prediction')
    # root.iconbitmap("Main_Logo.ico")

    # Text Boxes
    radius = Entry(root, width=30)
    radius.grid(row=0, column=1, padx=10, pady=10)

    texture = Entry(root, width=30)
    texture.grid(row=1, column=1, padx=10, pady=10)

    perimeter = Entry(root, width=30)
    perimeter.grid(row=2, column=1, padx=10, pady=10)

    area = Entry(root, width=30)
    area.grid(row=3, column=1, padx=10, pady=10)

    smoothness = Entry(root, width=30)
    smoothness.grid(row=4, column=1, padx=10, pady=10)

    compactness = Entry(root, width=30)
    compactness.grid(row=5, column=1, padx=10, pady=10)

    concavity = Entry(root, width=30)
    concavity.grid(row=6, column=1, padx=10, pady=10)

    concave_pt = Entry(root, width=30)
    concave_pt.grid(row=7, column=1, padx=10, pady=10)

    symmetry = Entry(root, width=30)
    symmetry.grid(row=8, column=1, padx=10, pady=10)

    fractal_dim = Entry(root, width=30)
    fractal_dim.grid(row=9, column=1, padx=10, pady=10)

    # Labels with Text Boxes
    radius_label = Label(root, text='Mean Radius')
    radius_label.grid(row=0, column=0, padx=10, pady=10)

    texture_label = Label(root, text='Mean Texture')
    texture_label.grid(row=1, column=0, padx=10, pady=10)

    perimeter_label = Label(root, text='Mean Perimeter')
    perimeter_label.grid(row=2, column=0, padx=10, pady=10)

    area_label = Label(root, text='Mean Area')
    area_label.grid(row=3, column=0, padx=10, pady=10)

    smoothness_label = Label(root, text='Mean Smoothness')
    smoothness_label.grid(row=4, column=0, padx=10, pady=10)

    compactness_label = Label(root, text='Mean Compactness')
    compactness_label.grid(row=5, column=0, padx=10, pady=10)

    concavity_label = Label(root, text='Mean Concavity')
    concavity_label.grid(row=6, column=0, padx=10, pady=10)

    concave_pt_label = Label(root, text='Mean Concave Points')
    concave_pt_label.grid(row=7, column=0, padx=10, pady=10)

    symmetry_label = Label(root, text='Mean Symmetry')
    symmetry_label.grid(row=8, column=0, padx=10, pady=10)

    fractal_dim_label = Label(root, text='Mean Fractal Dimen.')
    fractal_dim_label.grid(row=9, column=0, padx=10, pady=10)

    blank = Label(root, text='\t')
    blank.grid(row=10, column=0, padx=10, pady=5)

    def next_breast():
        root1 = Tk()
        root1.title('Breast Cancer Prediction')
        # root1.iconbitmap("Main_Logo.ico")

        # Text Boxes
        radius_err = Entry(root1, width=30)
        radius_err.grid(row=0, column=1, padx=10, pady=10)

        texture_err = Entry(root1, width=30)
        texture_err.grid(row=1, column=1, padx=10, pady=10)

        perimeter_err = Entry(root1, width=30)
        perimeter_err.grid(row=2, column=1, padx=10, pady=10)

        area_err = Entry(root1, width=30)
        area_err.grid(row=3, column=1, padx=10, pady=10)

        smoothness_err = Entry(root1, width=30)
        smoothness_err.grid(row=4, column=1, padx=10, pady=10)

        compactness_err = Entry(root1, width=30)
        compactness_err.grid(row=5, column=1, padx=10, pady=10)

        concavity_err = Entry(root1, width=30)
        concavity_err.grid(row=6, column=1, padx=10, pady=10)

        concave_pt_err = Entry(root1, width=30)
        concave_pt_err.grid(row=7, column=1, padx=10, pady=10)

        symmetry_err = Entry(root1, width=30)
        symmetry_err.grid(row=8, column=1, padx=10, pady=10)

        fractal_dim_err = Entry(root1, width=30)
        fractal_dim_err.grid(row=9, column=1, padx=10, pady=10)

        # Labels with Text Boxes
        radius_err_label = Label(root1, text='Radius Error')
        radius_err_label.grid(row=0, column=0, padx=10, pady=10)

        texture_err_label = Label(root1, text='Texture Error')
        texture_err_label.grid(row=1, column=0, padx=10, pady=10)

        perimeter_err_label = Label(root1, text='Perimeter Error')
        perimeter_err_label.grid(row=2, column=0, padx=10, pady=10)

        area_err_label = Label(root1, text='Area Error')
        area_err_label.grid(row=3, column=0, padx=10, pady=10)

        smoothness_err_label = Label(root1, text='Smoothness Error')
        smoothness_err_label.grid(row=4, column=0, padx=10, pady=10)

        compactness_err_label = Label(root1, text='Compactness Error')
        compactness_err_label.grid(row=5, column=0, padx=10, pady=10)

        concavity_err_label = Label(root1, text='Concavity Error')
        concavity_err_label.grid(row=6, column=0, padx=10, pady=10)

        concave_pt_err_label = Label(root1, text='Concave Pts. Error')
        concave_pt_err_label.grid(row=7, column=0, padx=10, pady=10)

        symmetry_err_label = Label(root1, text='Symmetry Error')
        symmetry_err_label.grid(row=8, column=0, padx=10, pady=10)

        fractal_dim_err_label = Label(root1, text='Fractal Dimen. Error'
                )
        fractal_dim_err_label.grid(row=9, column=0, padx=10, pady=10)

        blank = Label(root1, text='\t')
        blank.grid(row=10, column=0, padx=10, pady=5)

        def next_breast2():
            root2 = Tk()
            root2.title('Breast Cancer Prediction')
            # root2.iconbitmap("Main_Logo.ico")

            # Text Boxes
            radius_worst = Entry(root2, width=30)
            radius_worst.grid(row=0, column=1, padx=10, pady=10)

            texture_worst = Entry(root2, width=30)
            texture_worst.grid(row=1, column=1, padx=10, pady=10)

            perimeter_worst = Entry(root2, width=30)
            perimeter_worst.grid(row=2, column=1, padx=10, pady=10)

            area_worst = Entry(root2, width=30)
            area_worst.grid(row=3, column=1, padx=10, pady=10)

            smoothness_worst = Entry(root2, width=30)
            smoothness_worst.grid(row=4, column=1, padx=10, pady=10)

            compactness_worst = Entry(root2, width=30)
            compactness_worst.grid(row=5, column=1, padx=10, pady=10)

            concavity_worst = Entry(root2, width=30)
            concavity_worst.grid(row=6, column=1, padx=10, pady=10)

            concave_pt_worst = Entry(root2, width=30)
            concave_pt_worst.grid(row=7, column=1, padx=10, pady=10)

            symmetry_worst = Entry(root2, width=30)
            symmetry_worst.grid(row=8, column=1, padx=10, pady=10)

            fractal_dim_worst = Entry(root2, width=30)
            fractal_dim_worst.grid(row=9, column=1, padx=10, pady=10)

            # Labels with Text Boxes
            radius_worst_label = Label(root2, text='Worst Radius')
            radius_worst_label.grid(row=0, column=0, padx=10, pady=10)

            texture_worst_label = Label(root2, text='Worst Texture')
            texture_worst_label.grid(row=1, column=0, padx=10, pady=10)

            perimeter_worst_label = Label(root2, text='Worst Perimeter')
            perimeter_worst_label.grid(row=2, column=0, padx=10,
                    pady=10)

            area_worst_label = Label(root2, text='Worst Area')
            area_worst_label.grid(row=3, column=0, padx=10, pady=10)

            smoothness_worst_label = Label(root2,
                    text='Worst Smoothness')
            smoothness_worst_label.grid(row=4, column=0, padx=10,
                    pady=10)

            compactness_worst_label = Label(root2,
                    text='Worst Compactness')
            compactness_worst_label.grid(row=5, column=0, padx=10,
                    pady=10)

            concavity_worst_label = Label(root2, text='Worst Concavity')
            concavity_worst_label.grid(row=6, column=0, padx=10,
                    pady=10)

            concave_pt_worst_label = Label(root2,
                    text='Worst Concave Pts.')
            concave_pt_worst_label.grid(row=7, column=0, padx=10,
                    pady=10)

            symmetry_worst_label = Label(root2, text='Worst Symmetry')
            symmetry_worst_label.grid(row=8, column=0, padx=10, pady=10)

            fractal_dim_worst_label = Label(root2,
                    text='Worst Fractal Dimen.')
            fractal_dim_worst_label.grid(row=9, column=0, padx=10,
                    pady=10)

            blank = Label(root2, text='\t')
            blank.grid(row=10, column=0, padx=10, pady=5)

            # Function to Invoke Trained Model and Predict Results
            def predict_breast():
                rad = float(radius.get())
                text = float(texture.get())
                peri = float(perimeter.get())
                ar = float(area.get())
                smooth = float(smoothness.get())
                compact = float(compactness.get())
                concav = float(concavity.get())
                con_pt = float(concave_pt.get())
                sym = float(symmetry.get())
                f_dim = float(fractal_dim.get())
                rad_err = float(radius_err.get())
                text_err = float(texture_err.get())
                peri_err = float(perimeter_err.get())
                ar_err = float(area_err.get())
                smooth_err = float(smoothness_err.get())
                compact_err = float(compactness_err.get())
                concav_err = float(concavity_err.get())
                con_pt_err = float(concave_pt_err.get())
                sym_err = float(symmetry_err.get())
                f_dim_err = float(fractal_dim_err.get())
                rad_worst = float(radius_worst.get())
                text_worst = float(texture_worst.get())
                peri_worst = float(perimeter_worst.get())
                ar_worst = float(area_worst.get())
                smooth_worst = float(smoothness_worst.get())
                compact_worst = float(compactness_worst.get())
                concav_worst = float(concavity_worst.get())
                con_pt_worst = float(concave_pt_worst.get())
                sym_worst = float(symmetry_worst.get())
                f_dim_worst = float(fractal_dim_worst.get())

                # Clearing Text Boxes
                radius.delete(0, END)
                texture.delete(0, END)
                perimeter.delete(0, END)
                area.delete(0, END)
                smoothness.delete(0, END)
                compactness.delete(0, END)
                concavity.delete(0, END)
                concave_pt.delete(0, END)
                symmetry.delete(0, END)
                fractal_dim.delete(0, END)
                radius_err.delete(0, END)
                texture_err.delete(0, END)
                perimeter_err.delete(0, END)
                area_err.delete(0, END)
                smoothness_err.delete(0, END)
                compactness_err.delete(0, END)
                concavity_err.delete(0, END)
                concave_pt_err.delete(0, END)
                symmetry_err.delete(0, END)
                fractal_dim_err.delete(0, END)
                radius_worst.delete(0, END)
                texture_worst.delete(0, END)
                perimeter_worst.delete(0, END)
                area_worst.delete(0, END)
                smoothness_worst.delete(0, END)
                compactness_worst.delete(0, END)
                concavity_worst.delete(0, END)
                concave_pt_worst.delete(0, END)
                symmetry_worst.delete(0, END)
                fractal_dim_worst.delete(0, END)

                # Loading Fitted Model
                file = open('./Models/Breast_model_pkl', 'rb')
                model = pickle.load(file)
                file.close()

                input_data = (
                    rad,
                    text,
                    peri,
                    ar,
                    smooth,
                    compact,
                    concav,
                    con_pt,
                    sym,
                    f_dim,
                    rad_err,
                    text_err,
                    peri_err,
                    ar_err,
                    smooth_err,
                    compact_err,
                    concav_err,
                    con_pt_err,
                    sym_err,
                    f_dim_err,
                    rad_worst,
                    text_worst,
                    peri_worst,
                    ar_worst,
                    smooth_worst,
                    compact_worst,
                    concav_worst,
                    con_pt_worst,
                    sym_worst,
                    f_dim_worst,
                    )

                # Processing Input Data to Make it Suitable for Model
                np_array = np.asarray(input_data)
                data_reshaped = np_array.reshape(1, -1)

                prediction = model.predict(data_reshaped)

                # Displaying Result
                if prediction[0] == 0:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person does NOT have Breast Cancer'
                          , padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()
                else:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person does NOT have Breast Cancer'
                          , padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()

                root2.destroy()
                root1.destroy()

            submit_button = Button(root2, text='Predict',
                                   command=predict_breast, pady=10)
            submit_button.grid(
                row=11,
                column=0,
                columnspan=2,
                padx=10,
                pady=10,
                ipadx=78,
                )

            root2.mainloop()

        submit_button = Button(root1, text='Next',
                               command=next_breast2, pady=10)
        submit_button.grid(
            row=11,
            column=0,
            columnspan=2,
            padx=10,
            pady=10,
            ipadx=78,
            )

        root1.mainloop()

    submit_button = Button(root, text='Next', command=next_breast,
                           pady=10)
    submit_button.grid(
        row=11,
        column=0,
        columnspan=2,
        padx=10,
        pady=10,
        ipadx=78,
        )

    root.mainloop()


def parkinson():
    root = Tk()
    root.title('Parkinsons Disease Prediction')
    # root.iconbitmap("Main_Logo.ico")

    # Text Boxes
    fo = Entry(root, width=30)
    fo.grid(row=0, column=1, padx=10, pady=10)

    fhi = Entry(root, width=30)
    fhi.grid(row=1, column=1, padx=10, pady=10)

    flo = Entry(root, width=30)
    flo.grid(row=2, column=1, padx=10, pady=10)

    jit_per = Entry(root, width=30)
    jit_per.grid(row=3, column=1, padx=10, pady=10)

    jit_abs = Entry(root, width=30)
    jit_abs.grid(row=4, column=1, padx=10, pady=10)

    rap = Entry(root, width=30)
    rap.grid(row=5, column=1, padx=10, pady=10)

    ppq = Entry(root, width=30)
    ppq.grid(row=6, column=1, padx=10, pady=10)

    # Labels for Text Boxes
    fo_label = Label(root, text='Avg. Vocal Fundamental Freq.')
    fo_label.grid(row=0, column=0, padx=10, pady=10)

    fhi_label = Label(root, text='Max. Vocal Fundamental Freq.')
    fhi_label.grid(row=1, column=0, padx=10, pady=10)

    flo_label = Label(root, text='Min. Vocal Fundamental Freq.')
    flo_label.grid(row=2, column=0, padx=10, pady=10)

    jit_per_label = Label(root, text='Jitter (%)')
    jit_per_label.grid(row=3, column=0, padx=10, pady=10)

    jit_abs_label = Label(root, text='Jitter (Abs)')
    jit_abs_label.grid(row=4, column=0, padx=10, pady=10)

    rap_label = Label(root, text='MDVP : RAP')
    rap_label.grid(row=5, column=0, padx=10, pady=10)

    ppq_label = Label(root, text='MDVP : PPQ')
    ppq_label.grid(row=6, column=0, padx=10, pady=10)

    blank = Label(root, text='\t')
    blank.grid(row=7, column=0, padx=10, pady=5)

    def next_park():
        root1 = Tk()
        root1.title('Parkinsons Disease Prediction')
        # root1.iconbitmap("Main_Logo.ico")

        # Text Boxes
        jit_ddp = Entry(root1, width=30)
        jit_ddp.grid(row=0, column=1, padx=10, pady=10)

        mdvp_shim = Entry(root1, width=30)
        mdvp_shim.grid(row=1, column=1, padx=10, pady=10)

        mdvp_shimdb = Entry(root1, width=30)
        mdvp_shimdb.grid(row=2, column=1, padx=10, pady=10)

        shim_apq3 = Entry(root1, width=30)
        shim_apq3.grid(row=3, column=1, padx=10, pady=10)

        shim_apq5 = Entry(root1, width=30)
        shim_apq5.grid(row=4, column=1, padx=10, pady=10)

        shim_apq = Entry(root1, width=30)
        shim_apq.grid(row=5, column=1, padx=10, pady=10)

        shim_dda = Entry(root1, width=30)
        shim_dda.grid(row=6, column=1, padx=10, pady=10)

        # Labels for Text Boxes
        jit_ddp_label = Label(root1, text='Jitter : DDP')
        jit_ddp_label.grid(row=0, column=0, padx=10, pady=10)

        mdvp_shim_label = Label(root1, text='MDVP : Shimmer')
        mdvp_shim_label.grid(row=1, column=0, padx=10, pady=10)

        mdvp_shimdb_label = Label(root1, text='MDVP : Shimmer (dB)')
        mdvp_shimdb_label.grid(row=2, column=0, padx=10, pady=10)

        shim_apq3_label = Label(root1, text='Shimmer : APQ3')
        shim_apq3_label.grid(row=3, column=0, padx=10, pady=10)

        shim_apq5_label = Label(root1, text='Shimmer : APQ5')
        shim_apq5_label.grid(row=4, column=0, padx=10, pady=10)

        shim_apq_label = Label(root1, text='Shimmer : APQ')
        shim_apq_label.grid(row=5, column=0, padx=10, pady=10)

        shim_dda_label = Label(root1, text='Shimmer : DDA')
        shim_dda_label.grid(row=6, column=0, padx=10, pady=10)

        blank = Label(root1, text='\t')
        blank.grid(row=7, column=0, padx=10, pady=5)

        def next_park2():
            root2 = Tk()
            root2.title('Parkinsons Disease Prediction')
            # root2.iconbitmap("Main_Logo.ico")

            # Text Boxes
            nhr = Entry(root2, width=30)
            nhr.grid(row=0, column=1, padx=10, pady=10)

            hnr = Entry(root2, width=30)
            hnr.grid(row=1, column=1, padx=10, pady=10)

            rpde = Entry(root2, width=30)
            rpde.grid(row=2, column=1, padx=10, pady=10)

            dfa = Entry(root2, width=30)
            dfa.grid(row=3, column=1, padx=10, pady=10)

            spread1 = Entry(root2, width=30)
            spread1.grid(row=4, column=1, padx=10, pady=10)

            spread2 = Entry(root2, width=30)
            spread2.grid(row=5, column=1, padx=10, pady=10)

            d2 = Entry(root2, width=30)
            d2.grid(row=6, column=1, padx=10, pady=10)

            ppe = Entry(root2, width=30)
            ppe.grid(row=7, column=1, padx=10, pady=10)

            # Labels for Text Boxes
            nhr_label = Label(root2, text='NHR')
            nhr_label.grid(row=0, column=0, padx=10, pady=10)

            hnr_label = Label(root2, text='HNR')
            hnr_label.grid(row=1, column=0, padx=10, pady=10)

            rpde_label = Label(root2, text='RPDE')
            rpde_label.grid(row=2, column=0, padx=10, pady=10)

            dfa_label = Label(root2, text='DFA')
            dfa_label.grid(row=3, column=0, padx=10, pady=10)

            spread1_label = Label(root2, text='Spread 1')
            spread1_label.grid(row=4, column=0, padx=10, pady=10)

            spread2_label = Label(root2, text='Spread 2')
            spread2_label.grid(row=5, column=0, padx=10, pady=10)

            d2_label = Label(root2, text='D2')
            d2_label.grid(row=6, column=0, padx=10, pady=10)

            ppe_label = Label(root2, text='PPE')
            ppe_label.grid(row=7, column=0, padx=10, pady=10)

            blank = Label(root2, text='\t')
            blank.grid(row=10, column=0, padx=10, pady=5)

            # Function to Invoke Trained Model and Predict Results
            def predict_park():
                f = float(fo.get())
                fh = float(fhi.get())
                fl = float(flo.get())
                jper = float(jit_per.get())
                jabs = float(jit_abs.get())
                ra = float(rap.get())
                pp = float(ppq.get())
                dd = float(jit_ddp.get())
                sh = float(mdvp_shim.get())
                shdb = float(mdvp_shimdb.get())
                sha3 = float(shim_apq3.get())
                sha5 = float(shim_apq5.get())
                sha = float(shim_apq.get())
                d = float(shim_dda.get())
                nh = float(nhr.get())
                hn = float(hnr.get())
                rp = float(rpde.get())
                df = float(dfa.get())
                sp1 = float(spread1.get())
                sp2 = float(spread2.get())
                d2get = float(d2.get())
                ppeget = float(ppe.get())

                # Clearing Text Boxes
                fo.delete(0, END)
                fhi.delete(0, END)
                flo.delete(0, END)
                jit_per.delete(0, END)
                jit_abs.delete(0, END)
                rap.delete(0, END)
                ppq.delete(0, END)
                jit_ddp.delete(0, END)
                mdvp_shim.delete(0, END)
                mdvp_shimdb.delete(0, END)
                shim_apq3.delete(0, END)
                shim_apq5.delete(0, END)
                shim_apq.delete(0, END)
                shim_dda.delete(0, END)
                nhr.delete(0, END)
                hnr.delete(0, END)
                rpde.delete(0, END)
                dfa.delete(0, END)
                spread1.delete(0, END)
                spread2.delete(0, END)
                d2.delete(0, END)
                ppe.delete(0, END)

                # Loading Fitted Model & Standard Scaler
                file = open('./Models/Parkinsons_model_pkl', 'rb')
                model = pickle.load(file)
                file.close()
                file = open('./Models/Parkinsons_scaler_pkl', 'rb')
                scaler = pickle.load(file)
                file.close()

                input_data = (
                    f,
                    fh,
                    fl,
                    jper,
                    jabs,
                    ra,
                    pp,
                    dd,
                    sh,
                    shdb,
                    sha3,
                    sha5,
                    sha,
                    d,
                    nh,
                    hn,
                    rp,
                    df,
                    sp1,
                    sp2,
                    d2get,
                    ppeget,
                    )

                # Processing Input Data to Make it Suitable for Model
                np_array = np.asarray(input_data)
                data_reshaped = np_array.reshape(1, -1)
                std_data = scaler.transform(data_reshaped)

                prediction = model.predict(std_data)

                # Displaying Result
                if prediction[0] == 0:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person does NOT have Parkinsons Disease'
                          , padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()
                else:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person does NOT have Parkinsons Disease'
                          , padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()

                root2.destroy()
                root1.destroy()

            submit_button = Button(root2, text='Predict',
                                   command=predict_park, pady=10)
            submit_button.grid(
                row=11,
                column=0,
                columnspan=2,
                padx=10,
                pady=10,
                ipadx=78,
                )

            root2.mainloop()

        submit_button = Button(root1, text='Next', command=next_park2,
                               pady=10)
        submit_button.grid(
            row=11,
            column=0,
            columnspan=2,
            padx=10,
            pady=10,
            ipadx=78,
            )

        root1.mainloop()

    submit_button = Button(root, text='Next', command=next_park,
                           pady=10)
    submit_button.grid(
        row=8,
        column=0,
        columnspan=2,
        padx=10,
        pady=10,
        ipadx=78,
        )

    root.mainloop()


def kidney():
    root = Tk()
    root.title('Chronic Kidney Disease Prediction')
    # root.iconbitmap("Main_Logo.ico")

    # Text Boxes
    age = Entry(root, width=30)
    age.grid(row=0, column=1, padx=10, pady=10)

    bp = Entry(root, width=30)
    bp.grid(row=1, column=1, padx=10, pady=10)

    sg = Entry(root, width=30)
    sg.grid(row=2, column=1, padx=10, pady=10)

    al = Entry(root, width=30)
    al.grid(row=3, column=1, padx=10, pady=10)

    su = Entry(root, width=30)
    su.grid(row=3, column=1, padx=10, pady=10)

    rbc = Entry(root, width=30)
    rbc.grid(row=4, column=1, padx=10, pady=10)

    pc = Entry(root, width=30)
    pc.grid(row=5, column=1, padx=10, pady=10)

    pcc = Entry(root, width=30)
    pcc.grid(row=6, column=1, padx=10, pady=10)

    # Labels for Text Boxes
    age_label = Label(root, text='Age')
    age_label.grid(row=0, column=0, padx=10, pady=10)

    bp_label = Label(root, text='Blood Pressure')
    bp_label.grid(row=1, column=0, padx=10, pady=10)

    sg_label = Label(root, text='Specific Gravity')
    sg_label.grid(row=2, column=0, padx=10, pady=10)

    al_label = Label(root, text='Albumin (0-4)')
    al_label.grid(row=3, column=0, padx=10, pady=10)

    su_label = Label(root, text='Sugar (0-4)')
    su_label.grid(row=4, column=0, padx=10, pady=10)

    rbc_label = Label(root, text='RBC (0:Norm./1:Abnorm.)')
    rbc_label.grid(row=5, column=0, padx=10, pady=10)

    pc_label = Label(root, text='Pus Cell (0:Norm./1:Abnor.)')
    pc_label.grid(row=6, column=0, padx=10, pady=10)

    pcc_label = Label(root, text='Pus Clumps (0:Notpre./1:Pres.)')
    pcc_label.grid(row=6, column=0, padx=10, pady=10)

    blank = Label(root, text='\t')
    blank.grid(row=7, column=0, padx=10, pady=5)

    def next_kid():
        root1 = Tk()
        root1.title('Chronic Kidney Disease Prediction')
        # root1.iconbitmap("Main_Logo.ico")

        # Text Boxes
        ba = Entry(root1, width=30)
        ba.grid(row=0, column=1, padx=10, pady=10)

        bgr = Entry(root1, width=30)
        bgr.grid(row=1, column=1, padx=10, pady=10)

        bu = Entry(root1, width=30)
        bu.grid(row=2, column=1, padx=10, pady=10)

        sc = Entry(root1, width=30)
        sc.grid(row=3, column=1, padx=10, pady=10)

        sod = Entry(root1, width=30)
        sod.grid(row=4, column=1, padx=10, pady=10)

        pot = Entry(root1, width=30)
        pot.grid(row=5, column=1, padx=10, pady=10)

        hemo = Entry(root1, width=30)
        hemo.grid(row=6, column=1, padx=10, pady=10)

        # Labels for Text Boxes
        ba_label = Label(root1, text='Bacteria (0:Notpres./1:Pres.)')
        ba_label.grid(row=0, column=0, padx=10, pady=10)

        bgr_label = Label(root1, text='Blood Glucose Random')
        bgr_label.grid(row=1, column=0, padx=10, pady=10)

        bu_label = Label(root1, text='Blood Urea')
        bu_label.grid(row=2, column=0, padx=10, pady=10)

        sc_label = Label(root1, text='Serum Creatinine')
        sc_label.grid(row=3, column=0, padx=10, pady=10)

        sod_label = Label(root1, text='Sodium')
        sod_label.grid(row=4, column=0, padx=10, pady=10)

        pot_label = Label(root1, text='Potassium')
        pot_label.grid(row=5, column=0, padx=10, pady=10)

        hemo_label = Label(root1, text='Hemoglobin')
        hemo_label.grid(row=6, column=0, padx=10, pady=10)

        blank = Label(root1, text='\t')
        blank.grid(row=7, column=0, padx=10, pady=5)

        def next_kid2():
            root2 = Tk()
            root2.title('Chronic Kidney Disease Prediction')
            # root2.iconbitmap("Main_Logo.ico")

            # Text Boxes
            pcv = Entry(root2, width=30)
            pcv.grid(row=0, column=1, padx=10, pady=10)

            wbcc = Entry(root2, width=30)
            wbcc.grid(row=1, column=1, padx=10, pady=10)

            rbcc = Entry(root2, width=30)
            rbcc.grid(row=2, column=1, padx=10, pady=10)

            htn = Entry(root2, width=30)
            htn.grid(row=3, column=1, padx=10, pady=10)

            dm = Entry(root2, width=30)
            dm.grid(row=4, column=1, padx=10, pady=10)

            cad = Entry(root2, width=30)
            cad.grid(row=5, column=1, padx=10, pady=10)

            appet = Entry(root2, width=30)
            appet.grid(row=6, column=1, padx=10, pady=10)

            pe = Entry(root2, width=30)
            pe.grid(row=7, column=1, padx=10, pady=10)

            ane = Entry(root2, width=30)
            ane.grid(row=8, column=1, padx=10, pady=10)

            # Labels for Text Boxes
            pcv_label = Label(root2, text='Packed Cell Volume')
            pcv_label.grid(row=0, column=0, padx=10, pady=10)

            wbcc_label = Label(root2, text='WBC Count')
            wbcc_label.grid(row=1, column=0, padx=10, pady=10)

            rbcc_label = Label(root2, text='RBC Count')
            rbcc_label.grid(row=2, column=0, padx=10, pady=10)

            htn_label = Label(root2, text='Hypertension (0:No/1:Yes)')
            htn_label.grid(row=3, column=0, padx=10, pady=10)

            dm_label = Label(root2, text='Diabetes Mellitus (0:No/1:Yes)')
            dm_label.grid(row=4, column=0, padx=10, pady=10)

            cad_label = Label(root2, text='Coronary Artery Dis. (0:No/1:Yes)')
            cad_label.grid(row=5, column=0, padx=10, pady=10)

            appet_label = Label(root2, text='Appetite (0:Poor/1:Good)')
            appet_label.grid(row=6, column=0, padx=10, pady=10)

            pe_label = Label(root2, text='Pedal Edema (0:No/1:Yes)')
            pe_label.grid(row=7, column=0, padx=10, pady=10)

            ane_label = Label(root2, text='Anemia (0:No/1:Yes)')
            ane_label.grid(row=8, column=0, padx=10, pady=10)

            blank = Label(root2, text='\t')
            blank.grid(row=9, column=0, padx=10, pady=5)

            # Function to Invoke Trained Model and Predict Results
            def predict_kid():
                ag = float(age.get())
                bpget = float(bp.get())
                sgget = float(sg.get())
                alget = float(al.get())
                suget = float(su.get())
                rb = float(rbc.get())
                pcget = float(pc.get())
                pccget = float(pcc.get())
                baget = float(ba.get())
                bg = float(bgr.get())
                buget = float(bu.get())
                scget = float(sc.get())
                so = float(sod.get())
                po = float(pot.get())
                hem = float(hemo.get())
                pcvget = float(pcv.get())
                wbc = float(wbcc.get())
                rbccget = float(rbcc.get())
                ht = float(htn.get())
                dmget = float(dm.get())
                ca = float(cad.get())
                app = float(appet.get())
                peget = float(pe.get())
                an = float(ane.get())

                # Clearing Text Boxes
                age.delete(0, END)
                bp.delete(0, END)
                sg.delete(0, END)
                al.delete(0, END)
                su.delete(0, END)
                rbc.delete(0, END)
                pc.delete(0, END)
                pcc.delete(0, END)
                ba.delete(0, END)
                bgr.delete(0, END)
                bu.delete(0, END)
                sc.delete(0, END)
                sod.delete(0, END)
                pot.delete(0, END)
                hemo.delete(0, END)
                pcv.delete(0, END)
                wbcc.delete(0, END)
                rbcc.delete(0, END)
                htn.delete(0, END)
                dm.delete(0, END)
                cad.delete(0, END)
                appet.delete(0, END)
                pe.delete(0, END)
                ane.delete(0, END)

                # Loading Fitted Model
                file = open('./Models/Kidney_model_pkl', 'rb')
                model = pickle.load(file)
                file.close()

                input_data = (
                    ag,
                    bpget,
                    sgget,
                    alget,
                    suget,
                    rb,
                    pcget,
                    pccget,
                    baget,
                    bg,
                    buget,
                    scget,
                    so,
                    po,
                    hem,
                    pcvget,
                    wbc,
                    rbccget,
                    ht,
                    dmget,
                    ca,
                    app,
                    peget,
                    an,
                    )

                # Processing Input Data to Make it Suitable for Model
                np_array = np.asarray(input_data)
                data_reshaped = np_array.reshape(1, -1)

                prediction = model.predict(data_reshaped)

                # Displaying Result
                if prediction[0] == 0:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person does NOT have Chronic Kidney Disease'
                          , padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()
                else:
                    message = Tk()
                    message.title('Prediction Result')
                    Label(message,
                          text='The Person has Chronic Kidney Disease',
                          padx=10, pady=10).pack(padx=10, pady=10)
                    message.mainloop()

                root2.destroy()
                root1.destroy()

            submit_button = Button(root2, text='Predict',
                                   command=predict_kid, pady=10)
            submit_button.grid(
                row=11,
                column=0,
                columnspan=2,
                padx=10,
                pady=10,
                ipadx=78,
                )

            root2.mainloop()

        submit_button = Button(root1, text='Next', command=next_kid2,
                               pady=10)
        submit_button.grid(
            row=11,
            column=0,
            columnspan=2,
            padx=10,
            pady=10,
            ipadx=78,
            )

        root1.mainloop()

    submit_button = Button(root, text='Next', command=next_kid, pady=10)
    submit_button.grid(
        row=8,
        column=0,
        columnspan=2,
        padx=10,
        pady=10,
        ipadx=78,
        )

    root.mainloop()


# Function to Select Particular Disease
def further_op():
    what = selected.get()
    root.destroy()

    if what == 'Diabetes Prediction':
        diabetes()
    elif what == 'Heart Disease Prediction':
        heart()
    elif what == 'Breast Cancer Prediction':
        breast()
    elif what == 'Parkinsons Disease Prediction':
        parkinson()
    else:
        kidney()


blank = Label(root, text='\t')
blank.grid(row=3, column=2, padx=20, pady=5)
blank = Label(root, text='')
blank.grid(row=4, column=1)
Button(root, text='Open Detector', command=further_op, padx=7,
       pady=5).grid(row=4, column=2, padx=10, pady=15)

root.mainloop()