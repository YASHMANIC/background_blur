from flask import Flask, render_template, request, send_file, redirect, url_for
from main import BackgroundBlurrer
import os
import tempfile
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
            
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
            
        blur_strength = int(request.form.get("blur_strength", 31))
        
        # Generate unique filenames
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"input_{unique_id}.jpg"
        output_filename = f"output_{unique_id}.jpg"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        file.save(input_path)
        
        # Process image
        blurrer = BackgroundBlurrer()
        blurrer.blur_background(input_path, output_path, blur_strength)
        
        # Redirect to results page
        return redirect(url_for('results', 
                              input_img=input_filename, 
                              output_img=output_filename))
    
    return render_template("index.html")

@app.route("/results")
def results():
    input_img = request.args.get('input_img')
    output_img = request.args.get('output_img')
    
    if not input_img or not output_img:
        return redirect(url_for('index'))
    
    return render_template("results.html", 
                         input_img=input_img, 
                         output_img=output_img)

@app.route("/download/<filename>")
def download(filename):
    return send_file(
        os.path.join(app.config['RESULT_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)