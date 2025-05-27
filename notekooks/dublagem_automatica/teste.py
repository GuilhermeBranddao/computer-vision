import re
import os
import time
from datetime import datetime, timedelta
import translators as ts
from gtts import gTTS
from pydub import AudioSegment # <--- Adicionado para duração
from pydub.exceptions import CouldntDecodeError # <--- Para tratar erros do pydub
try:
    from playsound3 import playsound # <--- Adicionado para tocar áudio
except ImportError:
    print("AVISO: Biblioteca 'playsound' não encontrada. A reprodução de áudio será desabilitada.")
    print("Instale com: pip install playsound")
    playsound = None # Define playsound como None se não puder ser importado


# --- Configurações ---
# (Mantenha suas configurações existentes)
TRANSCRIPT_FILE_PATH = "data/transcricao_com_marcação_de_tempo_exemple.txt"
OUTPUT_AUDIO_DIR = "audio_dublado_pt"
TARGET_LANGUAGE = 'pt' # Português
SOURCE_LANGUAGE = 'en' # Inglês

# --- Funções Auxiliares ---
# (Mantenha parse_timestamp, get_transcript_content, segment_transcript, translate_segments)
# ... (seu código anterior para essas funções) ...

def parse_timestamp(ts_str):
    """Converte uma string de timestamp HH:MM:SS para um objeto timedelta."""
    try:
        h, m, s = map(int, ts_str.split(':'))
        return timedelta(hours=h, minutes=m, seconds=s)
    except ValueError:
        print(f"AVISO: Timestamp inválido encontrado e ignorado: {ts_str}")
        return None

def get_transcript_content(file_path):
    """Lê o conteúdo do arquivo de transcrição."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"ERRO: Arquivo de transcrição não encontrado em '{file_path}'.")
        return None

def segment_transcript(transcript_content):
    """Segmenta a transcrição."""
    if not transcript_content:
        return []
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2})(.*?(?=\d{2}:\d{2}:\d{2}|$))", re.DOTALL)
    matches = pattern.findall(transcript_content)
    segments = []
    for i, (timestamp_str, text_en_raw) in enumerate(matches):
        start_time_obj = parse_timestamp(timestamp_str)
        if start_time_obj is None: continue
        text_en = text_en_raw.strip()
        if not text_en: continue
        end_time_obj = None
        if i + 1 < len(matches):
            next_timestamp_str, _ = matches[i+1]
            end_time_obj = parse_timestamp(next_timestamp_str)
        segments.append({
            "id": i, "start_time_str": timestamp_str, "start_time_td": start_time_obj,
            "end_time_td": end_time_obj, "text_en": text_en, "text_pt": None,
            "audio_file_path": None, "audio_duration_sec": None # <--- Adicionado audio_duration_sec
        })
    return segments

def translate_segments(segments):
    """Traduz o texto de cada segmento."""
    print("\n--- Iniciando Tradução ---")
    for segment in segments:
        if segment["text_en"]:
            try:
                segment["text_pt"] = ts.translate_text(
                    segment["text_en"], translator='google',
                    from_language=SOURCE_LANGUAGE, to_language=TARGET_LANGUAGE
                )
                print(f"  Segmento {segment['id']} ({segment['start_time_str']}): Traduzido para PT.")
            except Exception as e:
                print(f"  ERRO na tradução do segmento {segment['id']}: {e}")
                segment["text_pt"] = f"[Erro na tradução: {segment['text_en']}]"
        else:
            segment["text_pt"] = ""
    print("--- Tradução Concluída ---")
    return segments


def generate_tts_for_segments(segments, output_dir):
    """
    Gera arquivos de áudio para os textos traduzidos, pulando se já existirem.
    Sempre tenta obter a duração do áudio.
    """
    print("\n--- Iniciando Geração/Verificação de Áudio (TTS) ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório de áudio criado: {output_dir}")

    for segment in segments:
        if not segment["text_pt"]: # Se não há texto traduzido (ex: erro na tradução ou texto original vazio)
            print(f"  Segmento {segment['id']}: Sem texto em português para gerar áudio.")
            segment["audio_file_path"] = None
            segment["audio_duration_sec"] = None
            continue

        # Define o nome esperado do arquivo de áudio
        safe_timestamp_str = segment['start_time_str'].replace(':', '_')
        audio_filename = f"segment_{segment['id']:03d}_{safe_timestamp_str}.mp3"
        audio_path = os.path.join(output_dir, audio_filename)
        segment["audio_file_path"] = audio_path # Define o caminho esperado

        if os.path.exists(audio_path):
            print(f"  Segmento {segment['id']}: Arquivo de áudio já existe em '{audio_path}'. Pulando geração TTS.")
            # Mesmo que o arquivo exista, tentamos obter/confirmar sua duração
            try:
                audio_segment_obj = AudioSegment.from_mp3(audio_path)
                segment["audio_duration_sec"] = len(audio_segment_obj) / 1000.0
                print(f"    Duração do áudio existente: {segment['audio_duration_sec']:.2f}s")
            except CouldntDecodeError:
                print(f"    AVISO: pydub não conseguiu decodificar o arquivo existente '{audio_path}'. Verifique FFmpeg/arquivo.")
                segment["audio_duration_sec"] = None
            except Exception as e_pydub:
                print(f"    AVISO: Erro ao obter duração do áudio existente {audio_filename} (pydub): {e_pydub}")
                segment["audio_duration_sec"] = None
        else:
            # Arquivo não existe, então geramos
            print(f"  Segmento {segment['id']}: Gerando áudio para '{audio_path}'...")
            try:
                tts = gTTS(text=segment["text_pt"], lang=TARGET_LANGUAGE, slow=False)
                tts.save(audio_path)
                print(f"    Áudio salvo em '{audio_path}'")

                # Obter duração do áudio recém-gerado
                try:
                    audio_segment_obj = AudioSegment.from_mp3(audio_path)
                    segment["audio_duration_sec"] = len(audio_segment_obj) / 1000.0
                    print(f"    Duração do áudio gerado: {segment['audio_duration_sec']:.2f}s")
                except CouldntDecodeError:
                    print(f"    AVISO: pydub não conseguiu decodificar o arquivo gerado '{audio_path}'. Verifique FFmpeg/arquivo.")
                    segment["audio_duration_sec"] = None
                except Exception as e_pydub:
                    print(f"    AVISO: Erro ao obter duração do áudio gerado {audio_filename} (pydub): {e_pydub}")
                    segment["audio_duration_sec"] = None

            except Exception as e_tts:
                print(f"  ERRO no TTS para segmento {segment['id']}: {e_tts}")
                segment["audio_file_path"] = None # Garante que não tentará tocar um arquivo que falhou na criação
                segment["audio_duration_sec"] = None

    print("--- Geração/Verificação de Áudio Concluída ---")
    return segments

def simulate_synchronized_playback(segments):
    """Simula a reprodução dos áudios sincronizados com os timestamps, com lógica de tempo aprimorada."""
    print("\n--- Iniciando Simulação de Playback Sincronizado com Áudio Real ---")
    if not segments:
        print("Nenhum segmento para simular.")
        return

    if playsound is None:
        print("Playback de áudio desabilitado pois a biblioteca 'playsound3' não foi carregada.")

    print("\nInstruções:")
    print("1. Prepare seu vídeo do YouTube e deixe-o pausado no início (00:00:00).")
    print("2. Quando estiver pronto, pressione Enter abaixo.")
    print("3. IMEDIATAMENTE após pressionar Enter, dê play no seu vídeo do YouTube.")
    print("O script tentará sincronizar o áudio com os timestamps do vídeo.")
    input("Pressione Enter para iniciar a simulação e, em seguida, dê play no vídeo...")

    initial_simulation_real_start_time = datetime.now() # Âncora de tempo real fixa
    print(f"Simulação iniciada às: {initial_simulation_real_start_time.strftime('%H:%M:%S.%f')}")

    for i, segment in enumerate(segments):
        target_segment_video_time_td = segment["start_time_td"] # Timestamp do vídeo para este segmento

        # Calcula o momento real absoluto em que este segmento deveria começar a tocar
        target_segment_real_trigger_time = initial_simulation_real_start_time + target_segment_video_time_td

        # Calcula quanto tempo esperar a partir de AGORA
        current_real_time = datetime.now()
        wait_duration_sec = (target_segment_real_trigger_time - current_real_time).total_seconds()

        if wait_duration_sec > 0:
            print(f"  Aguardando {wait_duration_sec:.3f}s para o segmento (timestamp vídeo: {segment['start_time_str']})...")
            time.sleep(wait_duration_sec)
        elif wait_duration_sec < 0:
            # Se wait_duration_sec for negativo, significa que já estamos atrasados para este segmento.
            # O script prosseguirá imediatamente.
            print(f"  AVISO: Atrasado {-wait_duration_sec:.3f}s para o segmento {segment['start_time_str']}. Tocando imediatamente.")

        # Este é o momento (aproximado) em que o áudio realmente começa, após o sleep (se houve)
        actual_audio_start_real_time = datetime.now()
        drift_from_target_sec = (actual_audio_start_real_time - target_segment_real_trigger_time).total_seconds()

        print(f"\n[{segment['start_time_str']} / Real: {actual_audio_start_real_time.strftime('%H:%M:%S.%f')} / Drift: {drift_from_target_sec:+.3f}s]")

        if segment["audio_file_path"] and os.path.exists(segment["audio_file_path"]) and playsound:
            print(f"  ▶️  TOCANDO ÁUDIO: {segment['audio_file_path']}")
            if segment.get("text_en"): print(f"     Texto Original: \"{segment['text_en'][:80]}...\"")
            if segment.get("text_pt"): print(f"     Texto Traduzido: \"{segment['text_pt'][:80]}...\"")
            if segment["audio_duration_sec"] is not None:
                print(f"     Duração Estimada do Áudio: {segment['audio_duration_sec']:.2f}s")
            
            try:
                # playsound é bloqueante por padrão, o script espera o áudio terminar.
                playsound(segment["audio_file_path"])
                print(f"  ⏹️  ÁUDIO TERMINADO: {segment['audio_file_path']}")
            # except PlaysoundException as pse: # Erro específico do playsound
            #      if pse: # Verificar se pse não é None (caso playsound seja None e PlaysoundException também)
            #         print(f"    ERRO ao tocar áudio com playsound3: {pse}")
            #         print(f"    Verifique se você tem um backend de áudio compatível instalado (ex: mpg123, GStreamer) ou se o arquivo não está corrompido.")
            except Exception as e_play: # Captura outras exceções potenciais
                print(f"    ERRO desconhecido ao tentar tocar áudio: {e_play}")

        elif not playsound and segment["audio_file_path"] and os.path.exists(segment["audio_file_path"]):
             print(f"  (Simulando) ▶️  TOCAR ÁUDIO: {segment['audio_file_path']} (playsound3 não disponível)")
             if segment["audio_duration_sec"] is not None:
                print(f"     Duração Estimada do Áudio: {segment['audio_duration_sec']:.2f}s")
                time.sleep(segment["audio_duration_sec"])
                print(f"  (Simulando) ⏹️  ÁUDIO TERMINADO: {segment['audio_file_path']}")
             else: print(f"     (Duração do áudio não pôde ser determinada)")
        elif not segment["audio_file_path"] or not os.path.exists(segment["audio_file_path"]):
            print(f"  ⚠️  Arquivo de áudio não encontrado ou não especificado para o segmento {segment['id']}.")

        # Informação sobre o segmento original no vídeo
        if segment.get("end_time_td"):
            duration_of_original_segment_video = (segment["end_time_td"] - segment["start_time_td"]).total_seconds()
            print(f"     (Segmento original no vídeo dura ~{duration_of_original_segment_video:.2f}s até o próximo timestamp)")
        elif segment["audio_duration_sec"]: # Último segmento
             print(f"     (Este é o último segmento. Áudio dublado dura ~{segment['audio_duration_sec']:.2f}s)")
    
    print("\n--- Simulação de Playback Concluída ---")

# --- Execução Principal ---
if __name__ == "__main__":
    print("### INICIANDO PROCESSO DE DUBLAGEM COM ÁUDIO REAL (SIMULADO) ###")
    transcript_data = get_transcript_content(TRANSCRIPT_FILE_PATH)
    if transcript_data:
        parsed_segments = segment_transcript(transcript_data)
        if not parsed_segments:
            print("Nenhum segmento válido encontrado na transcrição.")
        else:
            print(f"\n{len(parsed_segments)} segmentos parseados.")
            translated_segments = translate_segments(parsed_segments) # Tradução primeiro
            segments_with_audio = generate_tts_for_segments(translated_segments, OUTPUT_AUDIO_DIR) # Depois TTS/verificação
            simulate_synchronized_playback(segments_with_audio)
            print("\nProcesso finalizado.")
            print(f"Os arquivos de áudio estão em: '{os.path.abspath(OUTPUT_AUDIO_DIR)}'")
    else:
        print("Não foi possível carregar os dados da transcrição. Encerrando.")