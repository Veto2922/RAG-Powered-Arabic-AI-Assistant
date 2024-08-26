from flask import Flask, request, render_template_string
from models.predict_model import Predict
import traceback

app = Flask(__name__)

# Initialize the RAG model
predict = Predict()

# Define the template
template = """
انت مساعد ذكي للاشخاص الناطقين باللغة العربيه تساعدهم في معرفة اخر الاخبار في مجالات مثل التكنولوجيا والسياسة والرياضة من الملفات المتاح لك الاطلاع عليها ,وفي نهاية اجابتك اشكر المستخدم واسئله هل لديك اسئلة اخري 
 
knowledge you know:
{context}

Question: {question}

ماذا تفعل إذا لم تكن الإجابة مدرجة في السؤال أو السياق:
1. أخبر المستخدم أنك ليس لديك معلومات كافية للاجابة علي سواله
2. أخبر المستخدم أنك متخصص في فئات المقالات [الثقافة، والمال، والطب، والسياسة، والدين، والرياضة، والتكنولوجيا] فقط من المواقع الإلكترونية: الخليج، والعربية، وأخبارنا..
3. اسأل المستخدم إذا كان لديه المزيد من الأسئلة ليطرحها.
4. لا تذكر أي شيء عن السياق.

answer:
"""

# HTML template for rendering
html_template = """
<!doctype html>
<html>
<head><title>RAG-Powered Arabic AI Assistant</title></head>
<body>
    <h1>RAG-Powered Arabic AI Assistant</h1>
    <form method="POST">
        <label for="question">Ask a question:</label><br>
        <input type="text" id="question" name="question" required><br><br>
        <input type="submit" value="Get Answer">
    </form>
    {% if answer %}
        <h2>Answer:</h2>
        <div>{{ answer|safe }}</div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        try:
            answer = predict.get_answer(template, question)
        except Exception as e:
            answer = f"An error occurred: {traceback.format_exc()}"
    return render_template_string(html_template, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
