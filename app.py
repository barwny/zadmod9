import streamlit as st
import joblib
import pandas as pd
import json
import time
from dotenv import load_dotenv
import plotly.graph_objects as go
from langfuse.decorators import observe
from langfuse.openai import OpenAI as LangfuseOpenAI
import os

#zmienne środowiskowe
load_dotenv()

#Inicjalizacja klienta Langfuse (OpenAI przez Langfuse)
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = st.text_input("Wprowadz swój klucz API od OpenAI aby kontynuować", type="password")

if not openai_api_key:
    st.stop()

client = LangfuseOpenAI(api_key=openai_api_key)

#Funkcja z dekoratorem do śledzenia w Langfuse
@observe()
def parse_user_input_with_llm(system_prompt, user_input):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

#model regresji
model = joblib.load('marathon_regression_pipeline.pkl')

#UI
st.markdown(
    """
    <h3 style='text-align: center;
               color: #808080;
               font-size: 20px;
               font-weight: bold;
               font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               line-height: 1.4;
               margin-bottom: 20px;'>
        🧙‍♂️ Witaj! W tej wersji aplikacji postaram się dla ciebie wyczarować<br>
        szacowany czas ukończenia półmaratonu.<br>
        Dzięki tej informacji ocenisz swoją formę. 🏃
    </h3>
    """,
    unsafe_allow_html=True
)

user_input = st.text_area(
    "W poniższym polu wprowadź proszę kilka informacji (jak imię, płeć, wiek, czas na 5 km)📊",
    placeholder="np. Karolina, Kobieta, 25 lat, 22.5 min"
)

if st.button("Wyczaruj czas"):
    if not user_input.strip():
        st.warning("🧙‍♂️ Uzupełnij powyższe pole 👆")
        st.stop()

    system_prompt = """
    Wyodrębnij trzy wartości z wypowiedzi użytkownika:
    1. płeć (Kobieta lub Mężczyzna),
    2. wiek (liczba całkowita),
    3. czas na 5 km w minutach (może być liczba zmiennoprzecinkowa).

    Zwróć wynik w formacie JSON, np.:
    {"plec": "Mężczyzna", "wiek": 30, "czas_5km": 20.5}
    """

    
    try:
        response = parse_user_input_with_llm(system_prompt, user_input)
        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None
            st.error("Błąd dekodowania JSON.")

        if data is None or not isinstance(data, dict):
            st.error("Nieprawidłowe dane zwrócone z API.")
            st.stop()

        plec = data.get("plec")
        wiek = int(data.get("wiek")) if data.get("wiek") is not None else None
        czas_5km = float(data.get("czas_5km")) if data.get("czas_5km") is not None else None

        
        if czas_5km < 20 or czas_5km > 60:
            st.warning("🧙‍♂️ Czas na 5 km musi być w zakresie 20-60 minut.")
            st.stop()

        if plec not in ["Kobieta", "Mężczyzna"] or not isinstance(wiek, int) or not isinstance(czas_5km, (int, float)):
            st.error("🧙‍♂️ Nie mogę rozpoznać wprowadzonych danych.")
            st.stop()

        if wiek < 18 or wiek > 100:
            st.warning("🧙‍♂️ Moje czary działają tylko dla wieku 18-100 lat.")
            st.stop()

        if wiek <= 0 or czas_5km <= 0:
            st.error("🧙‍♂️ Dane muszą być większe niż zero.")
            st.stop()

        
        Płeć = 1 if plec.lower().startswith("m") else 0
        Rocznik = 2025 - wiek
        km5_Czas = czas_5km * 60

        features = pd.DataFrame([[Płeć, Rocznik, km5_Czas]],
                                columns=['Płeć', 'Rocznik', '5_km_czas'])

        
        predicted_time = model.predict(features)
        total_seconds = int(predicted_time[0])

        formatted_time = f"{int(total_seconds // 3600):02}:{int((total_seconds % 3600) // 60):02}:{int(total_seconds % 60):02}"

        
        progress_text = "Czaruję..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        
        if 3600 <= total_seconds < 5400:
            st.success(f"🧙‍♂️ Twój szacowany czas ukończenia półmaratonu: {formatted_time}. To profesjonalny wynik!")
        elif 5400 <= total_seconds <= 7200:
            st.success(f"🧙‍♂️ Twój szacowany czas ukończenia półmaratonu: {formatted_time}. Mega wynik!")
        elif 7200 <= total_seconds <= 9000:
            st.success(f"🧙‍♂️ Twój szacowany czas ukończenia półmaratonu: {formatted_time}. Świetny wynik!")
        elif 9000 <= total_seconds <= 10800:
            st.success(f"🧙‍♂️ Twój szacowany czas ukończenia półmaratonu: {formatted_time}. Dobry wynik!")
        else:
            st.success(f"🧙‍♂️ Twój szacowany czas ukończenia półmaratonu: {formatted_time}. Trzeba poprawić formę - zacznij trenować!")

        
        dystansy = ['5 km', '10 km', 'Półmaraton']
        czasy = [czas_5km, 2 * czas_5km, total_seconds / 60]

        fig = go.Figure(data=[go.Bar(name='Czasy', x=dystansy, y=czasy, marker_color='indianred')])
        fig.update_layout(
            title='Szacowany Czas na Różnych Dystansach',
            xaxis_title='Dystans',
            yaxis_title='Czas (minuty)',
        )
        st.plotly_chart(fig)

        
        st.sidebar.header("📊 Dane rozpoznane")
        if data:
            st.sidebar.json(data)
        else:
            st.sidebar.write("Brak danych do wyświetlenia.")

    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
