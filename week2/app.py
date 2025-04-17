import gradio as gr
import pandas as pd
import pickle

# Modell laden
model_filename = "apartment_price_model.pkl"  # Falls du das Modell gespeichert hast
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

def predict_price(rooms, area, postalcode, emp_density):
    input_data = pd.DataFrame([[rooms, area, postalcode, emp_density]],
                              columns=["rooms", "area", "postalcode", "emp_density"])
    prediction = model.predict(input_data)[0]
    return f"Geschätzter Mietpreis: CHF {prediction:.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Zimmeranzahl"),
        gr.Number(label="Wohnfläche in m²"),
        gr.Number(label="Postleitzahl"),
        gr.Number(label="Beschäftigungsdichte"),
    ],
    outputs="text",
    title="Wohnungspreis-Vorhersage",
    description="Gib die Wohnungsdetails ein, um den geschätzten Mietpreis zu erhalten."
)

demo.launch()
