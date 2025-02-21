import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import ffmpeg
import shutil
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib.pyplot as plt

def save_uploaded_file(uploaded_file, save_path):
    """Save uploaded file to a specified path."""
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

import os

def extract_audio(video_path, output_wav, duration="01:00:00"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    command = f'ffmpeg -i "{video_path}" -t {duration} -q:a 0 -map a "{output_wav}" -y'
    os.system(command)

    if not os.path.exists(output_wav):
        raise FileNotFoundError("Audio extraction failed! FFmpeg may not be working correctly.")

    return output_wav

def process_audio(input_wav, output_wav="processed_audio.wav", target_sample_rate=16000):
    """Convert stereo to mono and downsample audio."""
    audio = AudioSegment.from_wav(input_wav)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
    audio.export(output_wav, format="wav")
    return output_wav

def apply_energy_threshold(audio_wav, output_wav="filtered_audio.wav"):
    """Apply energy-based thresholding for excitement detection."""
    sample_rate, audio_data = wavfile.read(audio_wav)
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    energy = np.square(audio_data)
    threshold = 0.7 * np.max(energy)
    filtered_audio = np.where(energy > threshold, audio_data, 0)
    wavfile.write(output_wav, sample_rate, (filtered_audio * 32767).astype(np.int16))
    return output_wav

def extract_highlights(video_path, audio_path, output_folder, final_output_video):
    """Extract highlight clips based on high-energy moments."""
    os.makedirs(output_folder, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    threshold = 0.8 * np.max(energy)
    loud_moments = np.where(energy > threshold)[0] * 512 / sr
    merged_intervals = []
    threshold_gap, clip_duration = 5, 10
    if len(loud_moments) > 0:
        current_start, current_end = loud_moments[0], loud_moments[0] + clip_duration
        for time in loud_moments[1:]:
            if time - current_end <= threshold_gap:
                current_end = time + clip_duration
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = time, time + clip_duration
        merged_intervals.append((current_start, current_end))
    clip_paths = []
    for idx, (start, end) in enumerate(merged_intervals):
        clip_path = os.path.join(f"clip_{idx}.mp4")
        ffmpeg.input(video_path, ss=start, to=end).output(clip_path, vcodec="libx264", acodec="aac", preset="ultrafast").run(overwrite_output=True)
        clip_paths.append(clip_path)
    input_txt_path = os.path.join(output_folder, "input_list.txt")
    with open(input_txt_path, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")  # Ensure all clips are added

    ffmpeg.input(input_txt_path, format="concat", safe=0).output(final_output_video, vcodec="libx264", acodec="aac", preset="ultrafast").run(overwrite_output=True)
    return final_output_video, clip_paths

def main():
    st.title("Cricket Highlights Generator 🎬🏏")
    uploaded_file = st.file_uploader("Upload a cricket match video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        save_path = os.path.join("temp_videos", uploaded_file.name)
        os.makedirs("temp_videos", exist_ok=True)  # Ensure the directory exists
        
        # Write the file to the disk
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved successfully: {save_path}")

        # Now you can process the video with FFmpeg
        video_path = save_path
        
        st.video(video_path)
        output_audio = "extracted_audio.wav"
        processed_audio = "processed_audio.wav"
        filtered_audio = "filtered_audio.wav"
        
        highlight_folder = "highlight_clips"
        os.makedirs(highlight_folder, exist_ok=True) 
        
        final_highlight_video = "final_highlight.mp4"
        extract_audio(video_path, output_audio)
        process_audio(output_audio, processed_audio)
        apply_energy_threshold(processed_audio, filtered_audio)
        final_video, clips = extract_highlights(video_path, filtered_audio, highlight_folder, final_highlight_video)
        st.success("Highlight video generated successfully!")
        st.video(final_video)
        with open(final_video, "rb") as file:
            st.download_button("Download Highlights", file, file_name="cricket_highlights.mp4", mime="video/mp4")
        st.subheader("Extracted Highlight Clips")
        for clip in clips:
            st.video(clip)
if __name__ == "__main__":
    main()
