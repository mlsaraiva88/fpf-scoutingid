import streamlit as st
st.set_page_config(page_title="Relatório de Jogadores", layout="wide")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import base64
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Protocol
from enum import Enum
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# --- LOGIN ---
USERS = {
    "miguel": "senha123",
    "admin": "superadmin"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col_logo_left, col_logo_center, col_logo_right = st.columns([1, 2, 1])
    with col_logo_center:
        st.image("Logo Scouting.png", use_container_width=True, caption="ScoutingID")

    st.markdown("<h2 style='text-align: center; color: black;'>🔐 Login Obrigatório</h2>", unsafe_allow_html=True)

    col_form_left, col_form_center, col_form_right = st.columns([1, 1, 1])
    with col_form_center:
        username = st.text_input("Utilizador", key="login_username")
        password = st.text_input("Palavra-passe", type="password", key="login_password")

        if st.button("Entrar", key="login_button"):
            if USERS.get(username) == password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Credenciais incorretas.")
else:
    if st.sidebar.button("🔓 Terminar Sessão"):
        st.session_state.logged_in = False
        st.rerun()

    # --- Upload de Ficheiro Excel ---
    uploaded_file = st.file_uploader("📅 Upload ficheiro Excel", type="xlsx")

    df = pd.DataFrame()
    metrics_start_col_index = None

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            if 'Ações Totais' in df.columns:
                metrics_start_col_index = df.columns.get_loc('Ações Totais')
                st.success(f"✅ Índice da coluna 'Ações Totais': {metrics_start_col_index}")
            else:
                st.warning("⚠️ O ficheiro foi carregado, mas a coluna 'Ações Totais' não foi encontrada.")
        except Exception as e:
            st.error(f"❌ Erro ao ler o ficheiro: {e}")
    else:
        st.info("⬆️ Carrega um ficheiro `.xlsx` para começar a análise.")

    if not df.empty and metrics_start_col_index is not None:
        def normalize_data(df_to_normalize, columns_to_normalize):
            df_normalized = df_to_normalize.copy()
            for col in columns_to_normalize:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val == min_val:
                    df_normalized[col] = 0
                else:
                    df_normalized[col] = ((df_normalized[col] - min_val) / (max_val - min_val)) * 100
            return df_normalized

        all_metrics_columns = df.columns[metrics_start_col_index:].tolist()

        st.title("ScoutingID")
        st.sidebar.image("Logo Scouting.png", use_container_width=True)
        st.sidebar.header("Filtros de Dados")
        
        # Filtro 1: Jogadores
        all_players = ['Todos'] + sorted(df['Jogador'].unique().tolist())
        selected_players_filter = st.sidebar.multiselect("Selecionar Jogador(es)", all_players, default=['Todos'], key="jogador_multiselect")
        filtered_df = df.copy() if 'Todos' in selected_players_filter else df[df['Jogador'].isin(selected_players_filter)]
        
        # Filtro 2: Clubes
        all_clubs_selections = ['Todos'] + sorted(filtered_df['Clube ou Seleção'].unique().tolist())
        selected_club_selection = st.sidebar.multiselect("Selecionar Clube ou Seleção", all_clubs_selections, default=['Todos'], key="clube_multiselect")
        filtered_df = filtered_df.copy() if 'Todos' in selected_club_selection else filtered_df[filtered_df['Clube ou Seleção'].isin(selected_club_selection)]
        
        # Filtro 3: Ano de Nascimento
        all_birth_selections = ['Todos'] + sorted(filtered_df['Ano de Nascimento'].unique().tolist())
        selected_birth_selection = st.sidebar.multiselect("Selecionar Ano de Nascimento", all_birth_selections, default=['Todos'], key="ano_multiselect")
        filtered_df = filtered_df.copy() if 'Todos' in selected_birth_selection else filtered_df[filtered_df['Ano de Nascimento'].isin(selected_birth_selection)]
        
        # Filtro 4: Posição
        all_positions_data = ['Todas'] + sorted(filtered_df['Posição'].astype(str).unique().tolist())
        selected_position_data = st.sidebar.multiselect("Filtrar por Posição do Jogador (Dados)", all_positions_data, default=['Todas'], key="posicao_multiselect")
        final_filtered_df = filtered_df.copy() if 'Todas' in selected_position_data else filtered_df[filtered_df['Posição'].isin(selected_position_data)]

        st.sidebar.subheader("Opções de Dados")
        normalization_option = st.sidebar.radio(
        "Normalização de Dados",
        ("Dados Brutos (por 90 minutos)", "Normalizado (0-100)"),
        key="normalizacao_radio"
        )

        if normalization_option == "Normalizado (0-100)":
            metrics_to_normalize = [col for col in all_metrics_columns if col in final_filtered_df.columns]    
            if not final_filtered_df.empty and metrics_to_normalize:
                final_filtered_df = normalize_data(final_filtered_df, metrics_to_normalize)
            elif final_filtered_df.empty:
                st.sidebar.warning("Nenhum dado para normalizar após os filtros.")
        # --- SISTEMA DE PONDERAÇÕES DE COMPETIÇÕES ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚖️ Ponderações de Competições")
        
        # Dicionário com as ponderações sugeridas
        default_competition_weights = {
            # Ligas Top
            'England. Premier League': 2.0,
            'Italy. Serie A': 1.8,
            'France. Ligue 1': 1.7,
            'Netherlands. Eredivisie': 1.6,
            'England. Championship': 1.5,
            'Germany. 2. Bundesliga': 1.4,
            'Saudi Arabia. Pro League': 1.4,
            'Qatar. Qatar Stars League': 1.3,
            'Portugal. Segunda Liga': 1.2,
            'Portugal. Liga 3': 1.0,
            'Portugal. Liga Revelação Sub 23': 0.9,
            
            # Competições Europeias de Clubes
            'Europe. UEFA Champions League': 3.0,
            'Europe. UEFA Europa League - Qualification': 2.0,
            'Europe. UEFA Youth League': 1.5,
            
            # Competições de Seleções
            'Europe. UEFA U21 Championship Qualification': 2.5,
            'World. U21 National Team Friendlies': 1.8,
            'Europe. U20 Elite League': 1.7,
            'World. U19 National Team Friendlies': 1.6,
            
            # Taças
            'Italy. Coppa Italia': 1.6,
            'Netherlands. Super Cup': 1.5,
            'England. Carabao Cup': 1.4,
            'Saudi Arabia. King\'s Cup': 1.3,
            'Qatar. QSL Cup': 1.2,
            
            # Amigáveis
            'World. Emirates Cup': 0.8,
            'World. Club Friendlies': 0.7,
        }
        
        # Inicializar os pesos na session_state se não existirem
        if 'competition_weights' not in st.session_state:
            st.session_state.competition_weights = default_competition_weights.copy()
        
        # Opção para ativar/desativar ponderações
        use_weights = st.sidebar.checkbox(
            "🎚️ Ativar Ponderações de Competições",
            value=False,
            help="Ative para aplicar pesos diferentes às competições nas métricas"
        )
        
        if use_weights:
            # Obter competições únicas presentes nos dados
            if not df.empty and 'Competition' in df.columns:
                unique_competitions = sorted(df['Competition'].unique().tolist())
                
                # Seletor de preset
                st.sidebar.subheader("📋 Configuração Rápida")
                preset_option = st.sidebar.radio(
                    "Escolha um preset:",
                    ["Padrão Sugerido", "Todas Iguais (1.0)", "Personalizado"],
                    key="preset_weights"
                )
                
                if preset_option == "Todas Iguais (1.0)":
                    st.session_state.competition_weights = {comp: 1.0 for comp in unique_competitions}
                elif preset_option == "Padrão Sugerido":
                    # Aplicar pesos padrão, 1.0 para competições não listadas
                    st.session_state.competition_weights = {
                        comp: default_competition_weights.get(comp, 1.0) 
                        for comp in unique_competitions
                    }
                
                # Interface de edição
                with st.sidebar.expander("⚙️ Editar Ponderações", expanded=(preset_option == "Personalizado")):
                    st.markdown("**💡 Dica:** Peso > 1.0 = mais importante | Peso < 1.0 = menos importante")
                    
                    # Agrupar por tipo de competição para melhor organização
                    competition_categories = {
                        'Ligas Principais': [],
                        'Competições Europeias': [],
                        'Seleções': [],
                        'Taças': [],
                        'Amigáveis': [],
                        'Outras': []
                    }
                    
                    for comp in unique_competitions:
                        if 'Premier League' in comp or 'Serie A' in comp or 'Ligue 1' in comp or 'Eredivisie' in comp or 'Bundesliga' in comp or 'Segunda Liga' in comp or 'Liga 3' in comp or 'Liga Revelação' in comp or 'Pro League' in comp or 'Qatar Stars League' in comp or 'Championship' in comp:
                            competition_categories['Ligas Principais'].append(comp)
                        elif 'UEFA' in comp or 'Europa League' in comp or 'Champions League' in comp or 'Youth League' in comp:
                            competition_categories['Competições Europeias'].append(comp)
                        elif 'National Team' in comp or 'U21 Championship' in comp or 'U20 Elite' in comp or 'U19' in comp:
                            competition_categories['Seleções'].append(comp)
                        elif 'Cup' in comp or 'Coppa' in comp or 'Carabao' in comp or 'King' in comp or 'QSL Cup' in comp:
                            competition_categories['Taças'].append(comp)
                        elif 'Friendlies' in comp or 'Emirates' in comp:
                            competition_categories['Amigáveis'].append(comp)
                        else:
                            competition_categories['Outras'].append(comp)
                    
                    # Criar sliders por categoria
                    for category, comps in competition_categories.items():
                        if comps:
                            st.markdown(f"**{category}**")
                            for comp in sorted(comps):
                                current_weight = st.session_state.competition_weights.get(comp, 1.0)
                                
                                # Criar uma key única para cada slider
                                slider_key = f"weight_{comp}_{category}"
                                
                                new_weight = st.slider(
                                    comp,
                                    min_value=0.0,
                                    max_value=3.0,
                                    value=current_weight,
                                    step=0.1,
                                    key=slider_key,
                                    help=f"Peso atual: {current_weight}"
                                )
                                
                                st.session_state.competition_weights[comp] = new_weight
                
                # Botões de ação
                col_reset, col_export = st.sidebar.columns(2)
                
                with col_reset:
                    if st.button("🔄 Resetar", help="Voltar aos valores padrão"):
                        st.session_state.competition_weights = {
                            comp: default_competition_weights.get(comp, 1.0) 
                            for comp in unique_competitions
                        }
                        st.rerun()
                
                with col_export:
                    # Exportar configuração
                    import json
                    weights_json = json.dumps(st.session_state.competition_weights, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾",
                        data=weights_json,
                        file_name="ponderacoes_competicoes.json",
                        mime="application/json",
                        help="Exportar ponderações"
                    )
                
                # Mostrar resumo das ponderações ativas
                st.sidebar.markdown("---")
                st.sidebar.markdown("**📊 Resumo Rápido:**")
                
                # Calcular estatísticas
                weights_values = list(st.session_state.competition_weights.values())
                avg_weight = sum(weights_values) / len(weights_values)
                max_weight = max(weights_values)
                min_weight = min(weights_values)
                
                st.sidebar.metric("Média dos Pesos", f"{avg_weight:.2f}")
                
                col_min, col_max = st.sidebar.columns(2)
                with col_min:
                    st.metric("Mínimo", f"{min_weight:.1f}")
                with col_max:
                    st.metric("Máximo", f"{max_weight:.1f}")
                
                # Mostrar top 5 competições mais importantes
                top_5_weights = sorted(
                    st.session_state.competition_weights.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                st.sidebar.markdown("**🏆 Top 5 Mais Importantes:**")
                for comp, weight in top_5_weights:
                    comp_short = comp.split('. ')[-1] if '. ' in comp else comp
                    st.sidebar.text(f"{weight:.1f}x - {comp_short[:25]}")
                
            else:
                st.sidebar.warning("⚠️ Nenhuma competição encontrada nos dados")
        
        # --- FUNÇÃO PARA APLICAR PONDERAÇÕES ---
        def apply_competition_weights(df_input, weights_dict, metrics_columns):
            """
            Aplica ponderações às métricas com base na competição
            
            Args:
                df_input: DataFrame com os dados
                weights_dict: Dicionário com os pesos por competição
                metrics_columns: Lista de colunas de métricas para ponderar
            
            Returns:
                DataFrame com métricas ponderadas
            """
            if not use_weights or 'Competition' not in df_input.columns:
                return df_input
            
            df_weighted = df_input.copy()
            
            # Aplicar peso a cada métrica
            for idx, row in df_weighted.iterrows():
                competition = row['Competition']
                weight = weights_dict.get(competition, 1.0)
                
                # Multiplicar cada métrica pelo peso
                for metric in metrics_columns:
                    if metric in df_weighted.columns:
                        df_weighted.at[idx, metric] = df_weighted.at[idx, metric] * weight
            
            return df_weighted
        
        # --- INTEGRAÇÃO NO CÓDIGO PRINCIPAL ---
        # Aplicar ponderações ao final_filtered_df se ativado
        if use_weights and not final_filtered_df.empty:
            # Identificar colunas de métricas (a partir de 'Ações Totais')
            if metrics_start_col_index is not None:
                metrics_to_weight = df.columns[metrics_start_col_index:].tolist()
                
                # Aplicar ponderações
                final_filtered_df = apply_competition_weights(
                    final_filtered_df, 
                    st.session_state.competition_weights,
                    metrics_to_weight
                )
                
                # Mostrar indicador visual
                st.info("⚖️ **Ponderações Ativas:** As métricas foram ajustadas com base na importância das competições")
                
        st.subheader("📸 PlayerCards")
        PHOTO_FOLDER = "fotos_jogadores"
        if not final_filtered_df.empty:
            unique_players = sorted(final_filtered_df['Jogador'].unique().tolist())
            cards_html = ""
            for player_name in unique_players:
                player_data = final_filtered_df[final_filtered_df['Jogador'] == player_name]
                position_minutes = player_data.groupby('Posição')['Minutos'].sum()
                main_position = position_minutes.idxmax() if not position_minutes.empty and position_minutes.max() > 0 else "N/A"
                total_minutes = player_data['Minutos'].sum()
                total_goals = player_data['Golos'].sum() if 'Golos' in player_data.columns else 0
                total_assists = player_data['Assistências'].sum() if 'Assistências' in player_data.columns else 0
                photo_path_png = os.path.join(PHOTO_FOLDER, f"{player_name}.png")
                photo_path_jpg = os.path.join(PHOTO_FOLDER, f"{player_name}.jpg")
                image_data_b64 = None
                image_mime_type = ""
                if os.path.exists(photo_path_png):
                    with open(photo_path_png, "rb") as f:
                        image_data_b64 = base64.b64encode(f.read()).decode("ascii")
                    image_mime_type = "image/png"
                elif os.path.exists(photo_path_jpg):
                    with open(photo_path_jpg, "rb") as f:
                        image_data_b64 = base64.b64encode(f.read()).decode("ascii")
                    image_mime_type = "image/jpeg"
                if image_data_b64:
                    player_photo_html = f'<img src="data:{image_mime_type};base64,{image_data_b64}" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover; margin-bottom: 10px;">'
                else:
                    player_photo_html = '<div style="width: 80px; height: 80px; border-radius: 50%; background-color: #f0f2f6; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; font-size: 2em; color: #ccc;">⚽</div>'
                card_html = f""""
                <div style="
                    flex: 0 0 200px;
                    height: 280px;
                    border: 1px solid #e6e6e6;
                    border-radius: 8px;
                    padding: 10px;
                    margin-right: 10px;
                    box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
                    background-color: #ffffff;
                    text-align: center;
                    font-size: 0.85em;
                ">
                    {player_photo_html}
                    <h5 style="margin-top:0; margin-bottom: 5px; color: #333;">{player_name}</h5>
                    <p style="margin: 2px 0;"><strong>Posição:</strong> {main_position}</p>
                    <p style="margin: 2px 0;"><strong>Minutos:</strong> {total_minutes:.0f}</p>
                    <p style="margin: 2px 0;"><strong>Golos:</strong> {total_goals:.0f}</p>
                    <p style="margin: 2px 0;"><strong>Assists:</strong> {total_assists:.0f}</p>
                </div>
                """
                cards_html += card_html
            slider_container = f"""
            <div style="display: flex; overflow-x: auto; padding-bottom: 10px; margin-top: 10px;">
                {cards_html}
            </div>
            """
            st.markdown(slider_container, unsafe_allow_html=True)
        else:
            st.info("Nenhum jogador selecionado ou dados disponíveis para gerar os cartões de informação.")

    else:
        st.stop()


    # --- CONTEÚDO PROTEGIDO COMEÇA AQUI ---

    # Load the data
    @st.cache_data
    def load_data(file_path):
        # Use pd.read_excel for .xlsx files
        df = pd.read_excel(file_path)
    
        # Ensure numeric types for relevant columns starting from 'Ações Totais'
        # Find the index of 'Ações Totais'
        acoes_totais_col_index = df.columns.get_loc('Ações Totais')
        # Select all columns from 'Ações Totais' onwards
        numeric_cols = df.columns[acoes_totais_col_index:].tolist()
    
        for col in numeric_cols:
            # Convert to numeric, coerce errors to NaN, then fill NaN with 0.
            # This handles non-numeric data gracefully and assumes 0 for missing stats.
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
        return df
        

    # --- Data Normalization Function ---
    def normalize_data(df_to_normalize, columns_to_normalize):
        df_normalized = df_to_normalize.copy()
        for col in columns_to_normalize:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val == min_val:  # Avoid division by zero if all values are the same
                df_normalized[col] = 0
            else:
                df_normalized[col] = ((df_normalized[col] - min_val) / (max_val - min_val)) * 100
        return df_normalized

    # --- Função para Extrair Clube do Jogo ---
    def extract_club_from_game(game_string, default_clube="N/A"):
        """
        Extrai o nome do clube de uma string de jogo.
        Formato esperado: "Team1 - Team2 Score" (ex: "Lecce - Internazionale 0:2")
        Retorna o primeiro clube (equipa da casa) por padrão.
        """
        if pd.isna(game_string) or not isinstance(game_string, str):
            return default_clube

        try:
            # Remove o resultado (números após o último espaço)
            parts = game_string.split(' - ')
            if len(parts) >= 2:
                # Primeiro clube é o que vem antes do " - "
                return parts[0].strip()
        except:
            pass

        return default_clube

    # --- Função para Obter Clube dos Últimos 5 Jogos ---
    def get_most_common_club_from_last_games(player_data, num_games=5, col_data_jogo=None):
        """
        Obtém o clube mais frequente nos últimos N jogos do jogador.
        """
        if player_data.empty:
            return "N/A"

        # Ordenar por data se disponível
        df_sorted = player_data.copy()
        if col_data_jogo and col_data_jogo in df_sorted.columns:
            df_sorted = df_sorted.sort_values(col_data_jogo, ascending=False)

        # Pegar os últimos N jogos
        last_games = df_sorted.head(num_games)

        # Extrair clubes de cada jogo
        clubs = []
        for _, row in last_games.iterrows():
            if 'Jogo' in row.index:
                club = extract_club_from_game(row['Jogo'])
                clubs.append(club)

        # Encontrar o mais frequente
        if clubs:
            from collections import Counter
            most_common = Counter(clubs).most_common(1)
            if most_common:
                return most_common[0][0]

        return "N/A"

    # --- Metric Categorization ---
    # Identify columns from 'Ações Totais' onwards
    metrics_start_col_index = df.columns.get_loc('Ações Totais')
    all_metrics_columns = df.columns[metrics_start_col_index:].tolist()

    # Define categories based on typical football metrics
    # These lists are based on common football metric groupings, you can adjust them.
    # CORRESPONDÊNCIA EXATA COM O EXCEL - ATENÇÃO ÀS MAIÚSCULAS/MINÚSCULAS

    offensive_metrics = [

        'Golos', 'Assistências', 'Remates Totais', 'Remates à Baliza', 'xG',

        'Dribles Totais', 'Dribles Certos', 'Duelos Ofensivos Totais', 'Duelos Ofensivos Ganhos',

        'Toques na Área', 'Assistências para Remate', 'xA', 'Segundas Assistências'

    ]

    defensive_metrics = [

        'Intercepções', 'Recuperações Totais', 'Recuperações Meio Campo Adversário',

        'Carrinhos', 'Carrinhos Ganhos', 'Alívios', 'Duelos Defensivos Totais',

        'Duelos Defensivos Ganhos', 'Faltas', 'Duelos de Bola Parada Totais', 'Duelos de Bola Parada Ganhos'

    ]

    passing_metrics = [

        'Passes Totais', 'Passes Totais Certos', 'Passes Longos Totais', 'Passes Longos Certos',

        'Cruzamentos Totais', 'Cruzamentos Certos', 'Passes em Profundidade Totais',

        'Passes em Profundidade Certos', 'Passes para o Terço Final Totais',

        'Passes para o Terço Final Certos', 'Passes para a Grande Área Totais',

        'Passes para a Grande Área Certos', 'Passes recebidos',

        'Passes para o Frente Totais', 'Passes para o Frente Certos',

        'Passes para Trás Totais', 'Passes para Trás Certos', 'Pontapés de Baliza Curtos', 'Pontapés de Baliza Longos'

    ]

    general_metrics = [

        'Ações Totais', 'Ações Sucesso', 'Perdas Totais', 'Perdas Meio Campo', 'Duelos Aéreos Totais', 'Duelos Aéreos Ganhos',

        'Duelos Totais', 'Duelos Ganhos', 'Cartão Amarelo', 'Cartão Vermelho',

        'Foras de Jogo', 'Corridas Seguidas','Faltas Sofridas', 'Faltas'

    ]

    # Ensure all metrics are covered and no duplicates, and filter out any not in dataframe
    all_categorized_metrics = list(set(offensive_metrics + defensive_metrics + passing_metrics + general_metrics))
    all_categorized_metrics = [m for m in all_categorized_metrics if m in all_metrics_columns]

    # Create a mapping from metric name to its category
    metric_category_map = {}
    for metric in offensive_metrics:
        metric_category_map[metric] = 'Ofensiva'
    for metric in defensive_metrics:
        metric_category_map[metric] = 'Defensiva'
    for metric in passing_metrics:
        metric_category_map[metric] = 'Passe'
    for metric in general_metrics:
        metric_category_map[metric] = 'Geral'

    # Filter the map to only include metrics actually present in the DataFrame
    final_metric_category_map = {m: cat for m, cat in metric_category_map.items() if m in all_metrics_columns}

    # --- Metrics by Position Profile Dictionary ---
    # These are the "clusters" or "profiles" you want to define.
    # Ensure these metrics exist in your DataFrame columns, if not, they will be skipped
    # Mapping: Success = Ganhos/Certos
    metrics_by_position_profile = {
        '🧱 Centre-back (Defesa Central)': [
            # Com bola
            'Passes em Profundidade Certos',       # Passes progressivos (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes para o Terço Final Certos',   # Passes para último terço (com sucesso)
            'Passes Longos Certos',                # Passes longos (com sucesso)
            'Duelos Ofensivos Ganhos',             # Duelos ofensivos (com sucesso)
            'Duelos Aéreos Ganhos',               # Duelos aéreos ofensivos
            # Sem bola
            'Carrinhos Ganhos',                   # Desarmes (com sucesso)
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos Defensivos Ganhos',           # 1x1 defensivo (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos aéreos (com sucesso)
            'Intercepções',                        # Interceções
            'Recuperações Meio Campo Adversário',  # Recuperações em meio-campo ofensivo (com sucesso)
            'Faltas'                              # Faltas
        ],
        '🛡️ Full-back (Lateral)': [
            # Com bola
            'Passes em Profundidade Certos',       # Passes progressivos (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes para o Terço Final Certos',   # Passes para último terço (com sucesso)
            'Corridas Seguidas',                  # Corridas Seguidas (com sucesso)
            'Dribles Certos',                     # Dribles (com sucesso)
            'Duelos Ofensivos Ganhos',            # Duelos ofensivos (com sucesso)
            'Duelos Aéreos Ganhos',
            'Toques na Área',                     # Toques dentro da área
            'xA',                                 # xA
            'Assistências',                       # Assistências
            'Segundas Assistências',              # Segunda assistência
            'Assistências para Remate',            # Assistência para remate
            'Cruzamentos Certos',                 # Cruzamentos (com sucesso)
            'Remates à Baliza',                   # Remates (na baliza)
            'xG',                                 # xG
            # Sem bola
            'Carrinhos Ganhos',                   # Desarmes (com sucesso)
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos Defensivos Ganhos',           # 1x1 defensivo (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos aéreos (com sucesso)
            'Recuperações Meio Campo Adversário', # Recuperações em meio-campo ofensivo
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas
        ],
        '🧲 Defensive Midfielder (Médio Defensivo)': [
            # Com bola
            'Perdas Totais',                      # Perdas de bola
            'Passes Totais Certos',               # Passes (com sucesso)
            'Passes em Profundidade Certos',      # Passes progressivos (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes Longos Certos',               # Passes longos (com sucesso)
            'Passes para o Terço Final Certos',   # Passes no último terço (com sucesso)
            'Duelos Ofensivos Ganhos',
            'Duelos Aéreos Ganhos',                # Duelos ofensivos (com sucesso)
            'Dribles Certos',                     # Dribles (com sucesso)
            'Segundas Assistências',              # Segunda assistência
            # Sem bola
            'Recuperações Meio Campo Adversário', # Recuperações de bola no meio-campo ofensivo
            'Recuperações Totais',                # Recuperações Totais
            'Carrinhos Ganhos',                   # Desarmes (com sucesso)
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos aéreos (com sucesso)
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas por desarme
        ],
        '🧠 Central Midfielder (Médio Centro)': [
            # Com bola
            'Perdas Totais',                      # Perdas de bola
            'Corridas Seguidas',                  # Corridas Seguidas
            'Passes Totais Certos',               # Passes (com sucesso)
            'Passes em Profundidade Certos',      # Passes progressivos (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes Longos Certos',               # Passes longos (com sucesso)
            'Passes para o Terço Final Certos',   # Passes no último terço (com sucesso)
            'Duelos Ofensivos Ganhos',
            'Duelos Aéreos Ganhos',                # Duelos ofensivos (com sucesso)
            'Dribles Certos',                     # Dribles (com sucesso)
            'Assistências',                       # Assistências
            'Segundas Assistências',              # Segunda assistência
            'Assistências para Remate',           # Assistência para remate
            'xA',                                 # xA
            'Remates à Baliza',                   # Remates (à baliza)
            'Toques na Área',                     # Toques na Área
            'xG',                                 # xG
            'Golos',                              # Golos
            # Sem bola
            'Recuperações Meio Campo Adversário', # Recuperações de bola no meio-campo ofensivo
            'Recuperações Totais',                # Recuperações Totais
            'Carrinhos Ganhos',                   # Desarmes (com sucesso)
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos aéreo (com sucesso)
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas por desarme
        ],
        '🎯 Attacking Midfielder (Médio Ofensivo)': [
            # Com bola
            'Perdas Totais',                      # Perdas de bola
            'Corridas Seguidas',                  # Corridas Seguidas (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes para o Terço Final Certos',   # Passes no último terço (com sucesso)
            'Passes para a Grande Área Certos',   # Passes para a grande área (com sucesso)
            'Faltas Sofridas',                    # Faltas Sofridas
            'Duelos Ofensivos Ganhos',            # Duelos ofensivos (com sucesso)
            'Dribles Certos',                     # Dribles (com sucesso)
            'Toques na Área',                     # Toques dentro da área
            'Assistências para Remate',           # Assistência para remate na baliza
            'Assistências',                       # Assistências
            'Segundas Assistências',              # Segunda assistência
            'Assistências para Remate',           # Assistência para remate
            'Remates à Baliza',                   # Remates (na baliza)
            'xA',                                 # xA
            'xG',                                 # xG
            'Golos',                              # Golos
            # Sem bola
            'Recuperações Meio Campo Adversário', # Recuperações de bola no meio-campo ofensivo
            'Recuperações Totais',                # Recuperações Totais
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas
        ],
        '🪂 Winger (Extremo)': [
            # Com bola
            'Corridas Seguidas',                  # Corridas Seguidas (com sucesso)
            'Passes para o Frente Certos',        # Passes inteligentes / chave (com sucesso)
            'Passes para a Grande Área Certos',   # Passes para a grande área (com sucesso)
            'Passes para o Terço Final Certos',   # Passes no último terço (com sucesso)
            'Faltas Sofridas',                    # Faltas Sofridas
            'Duelos Ofensivos Ganhos',            # Duelos ofensivos (com sucesso)
            'Dribles Certos',                     # Dribles (com sucesso)
            'Toques na Área',                     # Toques dentro da área
            'Assistências para Remate',           # Assistência para remate na baliza
            'Assistências',                       # Assistências
            'Cruzamentos Certos',                 # Cruzamentos (com sucesso)
            'Segundas Assistências',              # Segunda assistência
            'Assistências para Remate',           # Assistência para remate
            'Remates à Baliza',                   # Remates (na baliza)
            'xA',                                 # xA
            'xG',                                 # xG
            'Golos',                              # Golos
            # Sem bola
            'Recuperações Meio Campo Adversário', # Recuperações de bola no meio-campo ofensivo
            'Recuperações Totais',                # Recuperações Totais
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos de Bola Parada Ganhos',       # Duelos na pressão (com sucesso)
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas
        ],
        '🎯 Striker (Avançado)': [
            # Com bola
            'Passes Totais Certos',               # Passes (com sucesso)
            'Corridas Seguidas',                  # Corridas Seguidas
            'Faltas Sofridas',                    # Faltas Sofridas
            'Duelos Ofensivos Ganhos',            # Duelos ofensivos (com sucesso)
            'Toques na Área',                     # Toques dentro da área
            'Assistências para Remate',           # Assistência para remate na baliza
            'Assistências',                       # Assistências
            'Assistências para Remate',           # Assistência para remate
            'Remates à Baliza',                   # Remates (na baliza)
            'Duelos de Bola Parada Totais',       # Duelos aéreos ofensivos
            'xA',                                 # xA
            'xG',                                 # xG
            'Golos',                              # Golos
            # Sem bola
            'Recuperações Meio Campo Adversário', # Recuperações de bola no meio-campo ofensivo
            'Recuperações Totais',                # Recuperações Totais
            'Duelos Defensivos Ganhos',           # Duelos defensivos (com sucesso)
            'Duelos de Bola Parada Ganhos',      # Duelos na pressão (com sucesso)
            'Intercepções',                        # Interceções
            'Faltas'                              # Faltas
        ]
    }

    # Filter metrics_by_position_profile to only include metrics that actually exist in the DataFrame
    for pos, metrics in metrics_by_position_profile.items():
        metrics_by_position_profile[pos] = [m for m in metrics if m in all_metrics_columns]


  
    st.subheader("Informação Geral por Jogador e Clube/Seleção")

    if not final_filtered_df.empty:
        general_info_data_rows = []

        # Get unique players from the filtered data
        unique_players_in_filtered_data = final_filtered_df['Jogador'].unique()

        for player in unique_players_in_filtered_data:
            player_df = final_filtered_df[final_filtered_df['Jogador'] == player].copy()
        
            # Group by 'Clube ou Seleção'
            grouped_by_club = player_df.groupby('Clube ou Seleção')

            for club_selection, group_df in grouped_by_club.groups.items(): # Iterate by group names
                # Get the actual DataFrame for this player and club combination
                current_player_club_df = player_df[player_df['Clube ou Seleção'] == club_selection]

                # Get unique positions for this player in this club
                unique_positions_in_club = current_player_club_df['Posição'].astype(str).unique()

                if len(unique_positions_in_club) == 0 or current_player_club_df.empty:
                    # Add a row for total if no specific position data for this club/player
                    row_data = {
                        'Jogador': player,
                        'Clube ou Seleção': club_selection,
                        'Posição (Nesta Posição no Clube)': 'N/A',
                        'Minutos Totais (Nesta Posição)': 0.0,
                        'Golos Totais (Nesta Posição)': 0.0,
                        'Assistências Totais (Nesta Posição)': 0.0,
                        'Média xG (Nesta Posição)': 0.0,
                        'Média xA (Nesta Posição)': 0.0,
                        'Ações Totais (Nesta Posição)': 0.0
                    }
                    general_info_data_rows.append(row_data)
                else:
                    for position in unique_positions_in_club:
                        # Filter data for the specific player, club, and position
                        player_club_position_df = current_player_club_df[current_player_club_df['Posição'].astype(str) == position]

                        total_minutes_pos = player_club_position_df['Minutos'].sum()
                        total_goals_pos = player_club_position_df['Golos'].sum() if 'Golos' in player_club_position_df.columns else 0
                        total_assists_pos = player_club_position_df['Assistências'].sum() if 'Assistências' in player_club_position_df.columns else 0
                        avg_xg_pos = player_club_position_df['xG'].mean() if 'xG' in player_club_position_df.columns else 0
                        avg_xa_pos = player_club_position_df['xA'].mean() if 'xA' in player_club_position_df.columns else 0
                        total_actions_pos = player_club_position_df['Ações Totais'].sum() if 'Ações Totais' in player_club_position_df.columns else 0

                        row_data = {
                            'Jogador': player,
                            'Clube ou Seleção': club_selection,
                            'Posição (Nesta Posição no Clube)': position,
                            'Minutos Totais (Nesta Posição)': total_minutes_pos,
                            'Golos Totais (Nesta Posição)': total_goals_pos,
                            'Assistências Totais (Nesta Posição)': total_assists_pos,
                            'Média xG (Nesta Posição)': avg_xg_pos,
                            'Média xA (Nesta Posição)': avg_xa_pos,
                            'Ações Totais (Nesta Posição)': total_actions_pos
                        }
                        general_info_data_rows.append(row_data)

        df_general_info = pd.DataFrame(general_info_data_rows)

        # Ensure numeric columns are numeric for styling and fillna 0 for numerical ops
        numeric_cols_for_info_table = [
            'Minutos Totais (Nesta Posição)', 'Golos Totais (Nesta Posição)', 
            'Assistências Totais (Nesta Posição)', 'Média xG (Nesta Posição)', 
            'Média xA (Nesta Posição)', 'Ações Totais (Nesta Posição)'
        ]
        for col in numeric_cols_for_info_table:
            df_general_info[col] = pd.to_numeric(df_general_info[col], errors='coerce').fillna(0)

        # Fill any remaining NaN values with 'N/A' for cleaner display *before* applying style
        df_general_info_filled = df_general_info.fillna('N/A')

        # Apply styling using background_gradient and format for display
        styled_df_general_info = df_general_info_filled.style.background_gradient(
            cmap='Greens',
            subset=['Minutos Totais (Nesta Posição)'] # Apply to minutes total per position
        ).format({
            'Minutos Totais (Nesta Posição)': "{:.0f}",
            'Golos Totais (Nesta Posição)': "{:.0f}",
            'Assistências Totais (Nesta Posição)': "{:.0f}",
            'Média xG (Nesta Posição)': "{:.2f}",
            'Média xA (Nesta Posição)': "{:.2f}",
            'Ações Totais (Nesta Posição)': "{:.0f}"
        })

        # Display the styled DataFrame
        st.dataframe(styled_df_general_info, hide_index=True, use_container_width=True)

    else:
        st.info("Nenhum dado disponível para gerar a informação geral dos jogadores.")

    # --- Tabbed Interface ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🌐 Radar Plot", "📊 ScoreID", "✅ Scatter Plot", "💎 Perfil de Posição", "🧬 Similaridade (Beta)", "🎯 KPI", "🎭 Arquétipos", "🏆 Forma Geral"])

    with tab1:
        st.header("Comparação de Jogadores (Radar Plot)")

        # MODIFICAÇÃO: df_comparison agora usa final_filtered_df
        # Filtra para os jogadores na df_final_filtered, mas mantém a estrutura de comparação
        df_comparison = final_filtered_df.copy()

        if df_comparison.empty:
            st.warning("Nenhum jogador presente nos dados filtrados. Ajuste os filtros na barra lateral.")
        else:
            # MODIFICAÇÃO: selected_highlight_player agora usa todos os jogadores únicos da df_comparison
            all_comparison_players_unique = sorted(df_comparison['Jogador'].unique().tolist())
        
            # Define um valor padrão mais robusto
            default_highlight_index = 0
            if 'Fábio Baldé' in all_comparison_players_unique:
                default_highlight_index = all_comparison_players_unique.index('Fábio Baldé')

            selected_highlight_player = st.selectbox(
                "Selecione um Jogador para Destacar",
                all_comparison_players_unique,
                index=default_highlight_index,
                key='radar_highlight_player'
            )

            st.write("Selecione as métricas para comparar os jogadores:")

            # --- New: Select metrics based on Position Profile ---
            position_profile_options = ['Manual (Selecionar Abaixo)'] + sorted(list(metrics_by_position_profile.keys()))
            selected_profile_for_radar = st.selectbox(
                "Preencher Métricas com Perfil de Posição",
                position_profile_options,
                key='select_profile_for_radar'
            )

            default_offensive = []
            default_defensive = []
            default_passing = []
            default_general = []

            if selected_profile_for_radar != 'Manual (Selecionar Abaixo)':
                profile_metrics = metrics_by_position_profile.get(selected_profile_for_radar, [])
                # Filtrar métricas que existem no DataFrame
                profile_metrics = [m for m in profile_metrics if m in all_metrics_columns]
                
                default_offensive = [m for m in profile_metrics if m in offensive_metrics and m in all_metrics_columns]
                default_defensive = [m for m in profile_metrics if m in defensive_metrics and m in all_metrics_columns]
                default_passing = [m for m in profile_metrics if m in passing_metrics and m in all_metrics_columns]
                default_general = [m for m in profile_metrics if m in general_metrics and m in all_metrics_columns]
            
                # Apenas adiciona 'Todas' se todas as métricas disponíveis da categoria estão no perfil
                available_offensive = [m for m in offensive_metrics if m in all_metrics_columns]
                available_defensive = [m for m in defensive_metrics if m in all_metrics_columns]
                available_passing = [m for m in passing_metrics if m in all_metrics_columns]
                available_general = [m for m in general_metrics if m in all_metrics_columns]
                
                if default_offensive and set(default_offensive) == set(available_offensive):
                    default_offensive = ['Todas']
                if default_defensive and set(default_defensive) == set(available_defensive):
                    default_defensive = ['Todas']
                if default_passing and set(default_passing) == set(available_passing):
                    default_passing = ['Todas']
                if default_general and set(default_general) == set(available_general):
                    default_general = ['Todas']


            # --- "Select All" functionality for metric multiselects ---
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.subheader("Métricas Ofensivas")
                all_off_metrics = [m for m in offensive_metrics if m in all_metrics_columns]
                selected_off_temp = st.multiselect(
                    "Selecione Métricas Ofensivas",
                    ['Todas'] + all_off_metrics,
                    default=default_offensive,
                    key=f'off_met_radar_{selected_profile_for_radar}'
                )
                if 'Todas' in selected_off_temp:
                    selected_offensive_metrics = all_off_metrics
                else:
                    selected_offensive_metrics = selected_off_temp

            with col2:
                st.subheader("Métricas Defensivas")
                all_def_metrics = [m for m in defensive_metrics if m in all_metrics_columns]
                selected_def_temp = st.multiselect(
                    "Selecione Métricas Defensivas",
                    ['Todas'] + all_def_metrics,
                    default=default_defensive,
                    key=f'def_met_radar_{selected_profile_for_radar}'
                )
                if 'Todas' in selected_def_temp:
                    selected_defensive_metrics = all_def_metrics
                else:
                    selected_defensive_metrics = selected_def_temp

            with col3:
                st.subheader("Métricas de Passe")
                all_pass_metrics = [m for m in passing_metrics if m in all_metrics_columns]
                selected_pass_temp = st.multiselect(
                    "Selecione Métricas de Passe",
                    ['Todas'] + all_pass_metrics,
                    default=default_passing,
                    key=f'pass_met_radar_{selected_profile_for_radar}'
                )
                if 'Todas' in selected_pass_temp:
                    selected_passing_metrics = all_pass_metrics
                else:
                    selected_passing_metrics = selected_pass_temp

            with col4:
                st.subheader("Métricas Gerais")
                all_gen_metrics = [m for m in general_metrics if m in all_metrics_columns]
                selected_gen_temp = st.multiselect(
                    "Selecione Métricas Gerais",
                    ['Todas'] + all_gen_metrics,
                    default=default_general,
                    key=f'gen_met_radar_{selected_profile_for_radar}'
                )
                if 'Todas' in selected_gen_temp:
                    selected_general_metrics = all_gen_metrics
                else:
                    selected_general_metrics = selected_gen_temp

            selected_metrics_for_chart = list(set(selected_offensive_metrics + selected_defensive_metrics + selected_passing_metrics + selected_general_metrics))

            if not selected_metrics_for_chart:
                st.info("Por favor, selecione pelo menos uma métrica para gerar o gráfico radar.")
            else:
                # Aggregate data for comparison players (e.g., mean)
                df_player_avg = df_comparison.groupby('Jogador')[selected_metrics_for_chart].mean().reset_index()

                # Calculate the average for all other players in the final_filtered_df
                # Exclude the selected_highlight_player from this average calculation
                other_players_df = final_filtered_df[final_filtered_df['Jogador'] != selected_highlight_player]
            
                # Ensure there are other players to calculate the average from
                if not other_players_df.empty and selected_metrics_for_chart:
                    average_other_players = other_players_df[selected_metrics_for_chart].mean()
                else:
                    average_other_players = pd.Series(0, index=selected_metrics_for_chart) # Default to 0 if no other players

                # Create the Radar Plot
                fig_radar = go.Figure()

                # Define colors for players: highlighted player is red, others are light grey
                player_colors = {player: 'lightgrey' for player in df_player_avg['Jogador'].unique()}
                if selected_highlight_player in player_colors:
                    player_colors[selected_highlight_player] = 'red' # Highlight the selected player

                for player in df_player_avg['Jogador'].unique():
                    player_data = df_player_avg[df_player_avg['Jogador'] == player]

                    # Get the values for the selected metrics for the current player
                    r_values = player_data[selected_metrics_for_chart].values[0].tolist()
                    theta_labels = selected_metrics_for_chart

                    # For a closed loop in radar chart, repeat the first value at the end
                    r_values_closed = r_values + [r_values[0]]
                    theta_labels_closed = theta_labels + [theta_labels[0]]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values_closed,
                        theta=theta_labels_closed,
                        fill='toself' if player == selected_highlight_player else 'none', # Only fill the highlighted player
                        name=player,
                        line_color=player_colors.get(player, 'grey'), # Use predefined colors, default to grey
                        opacity=0.8 if player == selected_highlight_player else 0.4, # Make highlighted player more opaque
                        hovertemplate=
                            "<b>Jogador:</b> %{name}<br>" +
                            "<b>Métrica:</b> %{theta}<br>" +
                            "<b>Valor:</b> %{r:.2f}<extra></extra>"
                    ))
            
                # Add the average line for other players
                if not average_other_players.empty and selected_metrics_for_chart:
                    avg_r_values = average_other_players[selected_metrics_for_chart].tolist()
                    avg_r_values_closed = avg_r_values + [avg_r_values[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=avg_r_values_closed,
                        theta=selected_metrics_for_chart + [selected_metrics_for_chart[0]],
                        fill='none',
                        name='Média dos Restantes Jogadores (Filtrados)', # Clarify that it's from filtered data
                        line_color='blue',
                        line_dash='dot',
                        opacity=0.6,
                        hovertemplate=
                            "<b>Média:</b> %{name}<br>" +
                            "<b>Métrica:</b> %{theta}<br>" +
                            "<b>Valor:</b> %{r:.2f}<extra></extra>"
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            # Dynamic range, slightly above max for non-normalized, fixed for normalized
                            range=[0, 100 if normalization_option == "Normalizado (0-100)" else max(df_player_avg[selected_metrics_for_chart].max().max() if not df_player_avg.empty else 0, average_other_players.max() if not average_other_players.empty else 0) * 1.1], 
                            showline=True,
                            linecolor='gray',
                            linewidth=1,
                            gridcolor='lightgray'
                        ),
                        angularaxis=dict(
                            showline=True,
                            linecolor='gray',
                            linewidth=1,
                            gridcolor='lightgray'
                        )
                    ),
                    showlegend=True,
                    title="Radar Plot de Comparação de Jogadores",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # --- Comparação de até 5 Jogadores ---
                st.subheader("📊 Comparação Detalhada de Jogadores")

                if not selected_metrics_for_chart:
                    st.info("Selecione métricas para comparar jogadores.")
                else:
                    # Seleção de até 5 jogadores para comparar
                    st.markdown("**Selecione até 7 jogadores para comparar:**")
                    col_select1, col_select2, col_select3, col_select4, col_select5, col_select6, col_select7  = st.columns(7)
                    
                    with col_select1:
                        player1 = st.selectbox(
                            "Jogador 1",
                            ['Nenhum'] + all_comparison_players_unique,
                            index=all_comparison_players_unique.index(selected_highlight_player) + 1 if selected_highlight_player in all_comparison_players_unique else 0,
                            key='compare_player1'
                        )
                    
                    with col_select2:
                        player2 = st.selectbox(
                            "Jogador 2",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player2'
                        )
                    
                    with col_select3:
                        player3 = st.selectbox(
                            "Jogador 3",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player3'
                        )
                    
                    with col_select4:
                        player4 = st.selectbox(
                            "Jogador 4",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player4'
                        )
                    
                    with col_select5:
                        player5 = st.selectbox(
                            "Jogador 5",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player5'
                        )
                    
                    with col_select6:
                        player6 = st.selectbox(
                            "Jogador 6",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player6'
                        )
                    with col_select7:
                        player7 = st.selectbox(
                            "Jogador 7",
                            ['Nenhum'] + all_comparison_players_unique,
                            key='compare_player7'
                        )

                    # Lista de jogadores selecionados
                    selected_compare_players = [p for p in [player1, player2, player3, player4, player5, player6, player7] if p != 'Nenhum']
                    
                    if not selected_compare_players:
                        st.info("👆 Selecione pelo menos um jogador para comparação.")
                    else:
                        # Função para obter top e bottom métricas
                        def get_top_bottom_metrics(player_name, metrics_list, top_n=10, bottom_n=5):
                            if player_name not in final_filtered_df['Jogador'].values:
                                return [], []
                            
                            player_data = final_filtered_df[final_filtered_df['Jogador'] == player_name][metrics_list].mean()
                            
                            # Calcular percentis para cada métrica
                            percentiles = {}
                            for metric in metrics_list:
                                if metric in final_filtered_df.columns:
                                    player_value = player_data.get(metric, 0)
                                    all_values = final_filtered_df[metric]
                                    if all_values.max() > 0:
                                        percentile = (all_values < player_value).sum() / len(all_values) * 100
                                        percentiles[metric] = (player_value, percentile)
                            
                            # Ordenar por percentil
                            sorted_metrics = sorted(percentiles.items(), key=lambda x: x[1][1], reverse=True)
                            
                            top_metrics = sorted_metrics[:top_n]
                            bottom_metrics = sorted_metrics[-bottom_n:][::-1]
                            
                            return top_metrics, bottom_metrics
                        
                        # Criar colunas para cada jogador
                        cols = st.columns(len(selected_compare_players))
                        
                        for idx, player_name in enumerate(selected_compare_players):
                            with cols[idx]:
                                # Card do jogador
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
                                    border-radius: 15px;
                                    padding: 20px;
                                    text-align: center;
                                    color: white;
                                    margin-bottom: 20px;
                                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                                ">
                                    <h3 style="margin: 0; font-size: 1.2em;">⚽ {player_name}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Obter top e bottom métricas
                                top_metrics, bottom_metrics = get_top_bottom_metrics(player_name, selected_metrics_for_chart, top_n=10, bottom_n=5)
                                
                                # Top 10 Métricas
                                st.markdown("**🏆 Top 10 Métricas**")
                                if top_metrics:
                                    for i, (metric, (value, percentile)) in enumerate(top_metrics, 1):
                                        # Cor baseada no percentil
                                        if percentile >= 80:
                                            color = "#4CAF50"
                                            emoji = "🟢"
                                        elif percentile >= 60:
                                            color = "#8BC34A"
                                            emoji = "🟡"
                                        else:
                                            color = "#FFC107"
                                            emoji = "🟠"
                                        
                                        st.markdown(f"""
                                        <div style="
                                            background: {color};
                                            border-radius: 8px;
                                            padding: 8px;
                                            margin: 4px 0;
                                            color: white;
                                            font-size: 0.85em;
                                        ">
                                            <b>{i}. {emoji} {metric}</b><br>
                                            <span style="font-size: 0.9em;">Valor: {value:.2f} | Percentil: {percentile:.0f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Sem dados")
                                
                                st.markdown("---")
                                
                                # Bottom 5 Métricas
                                st.markdown("**⚠️ Bottom 5 Métricas**")
                                if bottom_metrics:
                                    for i, (metric, (value, percentile)) in enumerate(bottom_metrics, 1):
                                        # Cor baseada no percentil
                                        if percentile <= 20:
                                            color = "#F44336"
                                            emoji = "🔴"
                                        elif percentile <= 40:
                                            color = "#FF5722"
                                            emoji = "🟠"
                                        else:
                                            color = "#FF9800"
                                            emoji = "🟡"
                                        
                                        st.markdown(f"""
                                        <div style="
                                            background: {color};
                                            border-radius: 8px;
                                            padding: 8px;
                                            margin: 4px 0;
                                            color: white;
                                            font-size: 0.85em;
                                        ">
                                            <b>{i}. {emoji} {metric}</b><br>
                                            <span style="font-size: 0.9em;">Valor: {value:.2f} | Percentil: {percentile:.0f}%</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Sem dados")
                                    
                    # --- EXECUTIVE SUMMARY ---
                        st.markdown("---")
                        st.subheader("📋 Executive Summary - Análise Comparativa")
                        
                        # Função para calcular score médio por categoria
                        def calculate_category_scores(player_name, metrics_list):
                            if player_name not in final_filtered_df['Jogador'].values:
                                return {}
                            
                            player_data = final_filtered_df[final_filtered_df['Jogador'] == player_name][metrics_list].mean()
                            
                            # Separar métricas por categoria
                            offensive_selected = [m for m in metrics_list if m in offensive_metrics]
                            defensive_selected = [m for m in metrics_list if m in defensive_metrics]
                            passing_selected = [m for m in metrics_list if m in passing_metrics]
                            general_selected = [m for m in metrics_list if m in general_metrics]
                            
                            scores = {}
                            
                            # Calcular percentil médio para cada categoria
                            for category_name, category_metrics in [
                                ('Ofensiva', offensive_selected),
                                ('Defensiva', defensive_selected),
                                ('Passe', passing_selected),
                                ('Geral', general_selected)
                            ]:
                                if category_metrics:
                                    percentiles = []
                                    for metric in category_metrics:
                                        if metric in final_filtered_df.columns:
                                            player_value = player_data.get(metric, 0)
                                            all_values = final_filtered_df[metric]
                                            if all_values.max() > 0:
                                                percentile = (all_values < player_value).sum() / len(all_values) * 100
                                                percentiles.append(percentile)
                                    
                                    if percentiles:
                                        scores[category_name] = {
                                            'avg_percentile': sum(percentiles) / len(percentiles),
                                            'metrics': category_metrics,
                                            'count': len(category_metrics)
                                        }
                            
                            return scores
                        
                        # Calcular scores para todos os jogadores selecionados
                        all_player_scores = {}
                        for player in selected_compare_players:
                            all_player_scores[player] = calculate_category_scores(player, selected_metrics_for_chart)
                        
                        # Encontrar o melhor jogador em cada categoria
                        categories = ['Ofensiva', 'Defensiva', 'Passe', 'Geral']
                        category_icons = {
                            'Ofensiva': '⚔️',
                            'Defensiva': '🛡️',
                            'Passe': '🎯',
                            'Geral': '📊'
                        }
                        
                        summary_bullets = []
                        
                        for category in categories:
                            # Encontrar jogador com melhor score nesta categoria
                            best_player = None
                            best_score = -1
                            
                            for player, scores in all_player_scores.items():
                                if category in scores:
                                    if scores[category]['avg_percentile'] > best_score:
                                        best_score = scores[category]['avg_percentile']
                                        best_player = player
                            
                            if best_player and best_score > 0:
                                metrics_list = all_player_scores[best_player][category]['metrics']
                                metrics_count = all_player_scores[best_player][category]['count']
                                
                                # Determinar cor do badge baseado no percentil
                                if best_score >= 75:
                                    badge_color = "#4CAF50"
                                    badge_text = "Excelente"
                                elif best_score >= 60:
                                    badge_color = "#8BC34A"
                                    badge_text = "Muito Bom"
                                elif best_score >= 50:
                                    badge_color = "#FFC107"
                                    badge_text = "Bom"
                                else:
                                    badge_color = "#FF9800"
                                    badge_text = "Acima da Média"
                                
                                summary_bullets.append({
                                    'category': category,
                                    'icon': category_icons[category],
                                    'player': best_player,
                                    'score': best_score,
                                    'metrics': metrics_list[:5],  # Mostrar até 5 métricas principais
                                    'metrics_count': metrics_count,
                                    'badge_color': badge_color,
                                    'badge_text': badge_text
                                })
                        
                        # Exibir summary
                        if summary_bullets:
                            for item in summary_bullets:
                                metrics_text = ", ".join(item['metrics'])
                                extra_metrics = item['metrics_count'] - len(item['metrics'])
                                if extra_metrics > 0:
                                    metrics_text += f" (+{extra_metrics} outras)"
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                                    border-left: 5px solid {item['badge_color']};
                                    border-radius: 10px;
                                    padding: 15px 20px;
                                    margin: 10px 0;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                ">
                                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                        <span style="font-size: 1.5em;">{item['icon']}</span>
                                        <h4 style="margin: 0; color: #2c3e50;">Categoria {item['category']}</h4>
                                        <span style="
                                            background: {item['badge_color']};
                                            color: white;
                                            padding: 4px 12px;
                                            border-radius: 20px;
                                            font-size: 0.85em;
                                            font-weight: bold;
                                            margin-left: auto;
                                        ">{item['badge_text']}</span>
                                    </div>
                                    <p style="margin: 5px 0; color: #34495e; font-size: 1.05em;">
                                        <strong style="color: {item['badge_color']};">🏆 {item['player']}</strong> demonstra o melhor desempenho 
                                        com <strong>{item['score']:.1f}%</strong> de percentil médio
                                    </p>
                                    <p style="margin: 5px 0; color: #7f8c8d; font-size: 0.95em;">
                                        <strong>📌 Métricas chave:</strong> {metrics_text}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Selecione jogadores e métricas para ver o resumo comparativo.")
                        
                        # Resumo geral - melhor jogador overall
                        st.markdown("---")
                        st.markdown("**🌟 Resumo Geral**")
                        
                        overall_scores = {}
                        for player in selected_compare_players:
                            if player in all_player_scores:
                                all_percentiles = []
                                for cat_scores in all_player_scores[player].values():
                                    all_percentiles.append(cat_scores['avg_percentile'])
                                
                                if all_percentiles:
                                    overall_scores[player] = sum(all_percentiles) / len(all_percentiles)
                        
                        if overall_scores:
                            best_overall = max(overall_scores, key=overall_scores.get)
                            best_overall_score = overall_scores[best_overall]
                            
                            # Ranking de todos os jogadores
                            sorted_players = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                            
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 15px;
                                padding: 20px;
                                color: white;
                                text-align: center;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                            ">
                                <h3 style="margin: 0 0 10px 0;">👑 Melhor Performance Global</h3>
                                <h2 style="margin: 10px 0; font-size: 2em;">{best_overall}</h2>
                                <p style="margin: 5px 0; font-size: 1.2em; opacity: 0.9;">
                                    Score Médio: {best_overall_score:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Ranking completo
                            st.markdown("<br>", unsafe_allow_html=True)
                            cols_ranking = st.columns(len(sorted_players))
                            
                            for idx, (player, score) in enumerate(sorted_players):
                                with cols_ranking[idx]:
                                    position = idx + 1
                                    if position == 1:
                                        medal = "🥇"
                                        color = "#FFD700"
                                    elif position == 2:
                                        medal = "🥈"
                                        color = "#C0C0C0"
                                    elif position == 3:
                                        medal = "🥉"
                                        color = "#CD7F32"
                                    else:
                                        medal = f"{position}º"
                                        color = "#95a5a6"
                                    
                                    st.markdown(f"""
                                    <div style="
                                        background: {color};
                                        border-radius: 10px;
                                        padding: 15px;
                                        text-align: center;
                                        color: white;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                                    ">
                                        <div style="font-size: 2em;">{medal}</div>
                                        <div style="font-size: 0.9em; font-weight: bold; margin: 5px 0;">{player}</div>
                                        <div style="font-size: 0.85em;">{score:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                                    
                # --- NEW RANKING TABLE ---
                st.subheader("Ranking Detalhado dos Jogadores para as Métricas Selecionadas")

                if not selected_metrics_for_chart or final_filtered_df.empty:
                    st.info("Selecione métricas e garanta que há dados filtrados para ver a tabela de rankings.")
                else:
                    max_players_in_ranking = final_filtered_df['Jogador'].nunique()
                    if max_players_in_ranking == 0:
                        st.warning("Não há jogadores nos dados filtrados para gerar o ranking.")
                    else:
                        # Seleção de múltiplos jogadores para destacar
                        all_available_players_for_highlight = sorted(final_filtered_df['Jogador'].unique())
                        players_to_highlight = st.multiselect(
                            "⭐ Selecione Jogadores para Destacar (Cor Dourada)",
                            options=all_available_players_for_highlight,
                            default=[],
                            key='multi_highlight_ranking',
                            help="Selecione um ou mais jogadores para destacá-los na tabela com cor dourada"
                        )
                        
                        num_ranks_to_display = st.slider(
                            "Número de Lugares do Ranking a Exibir",
                            min_value=1,
                            max_value=10,
                            value=min(10,max_players_in_ranking), # Default to 10 or max available
                            key='num_ranks_slider'
                        )

                        # Initialize ranking_table_data with 'Categoria' as the first column
                        ranking_table_data = {'Categoria': [], 'Métrica': []}
                    
                        # Add highlighted player's rank column if a player is selected
                        if selected_highlight_player != 'Nenhum':
                            ranking_table_data[f'Rank de {selected_highlight_player}'] = []

                        player_metric_means = {}
                        for metric in selected_metrics_for_chart:
                            if metric in final_filtered_df.columns:
                                temp_df = final_filtered_df.groupby('Jogador')[metric].mean().reset_index()
                                if not temp_df.empty and temp_df[metric].nunique() > 0: # Ensure there's data and variation for ranking
                                    temp_df['Rank'] = temp_df[metric].rank(ascending=False, method='min')
                                    player_metric_means[metric] = temp_df
                                else:
                                    player_metric_means[metric] = pd.DataFrame(columns=['Jogador', metric, 'Rank']) # Empty df for this metric
                            else:
                                player_metric_means[metric] = pd.DataFrame(columns=['Jogador', metric, 'Rank']) # Empty df if metric not found

                        # Dynamically create rank columns based on slider value
                        for i in range(1, num_ranks_to_display + 1):
                            ranking_table_data[f'{i}º Lugar'] = []

                        overall_averages_for_ranking = final_filtered_df[selected_metrics_for_chart].mean()

                        for metric in selected_metrics_for_chart:
                            # Populate the new 'Categoria' column
                            ranking_table_data['Categoria'].append(final_metric_category_map.get(metric, 'Outra'))
                            ranking_table_data['Métrica'].append(metric)
                        
                            # Populate highlighted player's rank with medals
                            if selected_highlight_player != 'Nenhum':
                                highlight_player_rank_str = "N/A"
                                if metric in player_metric_means and not player_metric_means[metric].empty and selected_highlight_player in player_metric_means[metric]['Jogador'].values:
                                    raw_rank = int(player_metric_means[metric][player_metric_means[metric]['Jogador'] == selected_highlight_player]['Rank'].iloc[0])
                                    if raw_rank == 1:
                                        highlight_player_rank_str = "🥇 1º"
                                    elif raw_rank == 2:
                                        highlight_player_rank_str = "🥈 2º"
                                    elif raw_rank == 3:
                                        highlight_player_rank_str = "🥉 3º"
                                    else:
                                        highlight_player_rank_str = f"{raw_rank}º"
                                ranking_table_data[f'Rank de {selected_highlight_player}'].append(highlight_player_rank_str)

                            current_metric_ranks = {rank: [] for rank in range(1, num_ranks_to_display + 1)}
                        
                            df_ranked_players_for_metric = player_metric_means.get(metric, pd.DataFrame())

                            if not df_ranked_players_for_metric.empty:
                                df_ranked_players_for_metric = df_ranked_players_for_metric[df_ranked_players_for_metric['Rank'] <= num_ranks_to_display]

                                for idx, row in df_ranked_players_for_metric.iterrows():
                                    player_name = row['Jogador']
                                    player_rank = int(row['Rank'])

                                    # Get the player's actual value for this metric
                                    player_metric_value = row[metric]

                                    # Get the overall average for this metric
                                    overall_average_for_metric = overall_averages_for_ranking.get(metric, 0)

                                    color_emoji = ""
                                    if player_metric_value is not None and overall_average_for_metric is not None:
                                        # Only add emoji if there's a meaningful comparison (i.e., not all zeros if overall average is zero)
                                        if overall_average_for_metric != 0:
                                            if player_metric_value >= overall_average_for_metric:
                                                color_emoji = "🟢 "
                                            else:
                                                color_emoji = "🔴 "
                                        elif player_metric_value > 0 and overall_average_for_metric == 0: 
                                            # If overall average is 0 but player has a positive value, consider it "good"
                                            color_emoji = "🟢 "
                                        # If both are 0, no emoji (neutral)

                                    base_display_name = f"{color_emoji}{player_name}"

                                    if player_name == selected_highlight_player:
                                        display_name = f"🔎 {base_display_name}"
                                    else:
                                        display_name = base_display_name
                                
                                    if player_rank <= num_ranks_to_display:
                                        current_metric_ranks[player_rank].append(display_name)

                            for r_col in range(1, num_ranks_to_display + 1):
                                # Modificar para destacar jogadores selecionados em dourado
                                highlighted_players_in_rank = []
                                normal_players_in_rank = []
                                
                                for player_display in current_metric_ranks[r_col]:
                                    # Extrair o nome real do jogador (remover emojis e símbolos)
                                    player_name_clean = player_display.replace("🟢 ", "").replace("🔴 ", "").replace("🔎 ", "").strip()
                                    
                                    if player_name_clean in players_to_highlight:
                                        # Destacar com emoji de estrela dourada
                                        highlighted_players_in_rank.append(f"⭐ {player_display}")
                                    else:
                                        normal_players_in_rank.append(player_display)
                                
                                # Combinar jogadores destacados primeiro, depois os normais
                                all_players_for_rank = highlighted_players_in_rank + normal_players_in_rank
                                ranking_table_data[f'{r_col}º Lugar'].append(', '.join(all_players_for_rank) if all_players_for_rank else 'N/A')
                            
                    df_rank_display = pd.DataFrame(ranking_table_data)

                    # Aplicar estilo com destaque dourado para jogadores selecionados
                    def highlight_gold_players(row):
                        """Destaca células que contêm jogadores marcados com ⭐"""
                        styles = []
                        for val in row:
                            if isinstance(val, str) and '⭐' in val:
                                styles.append('background-color: #FFD700; color: black; font-weight: bold;')
                            else:
                                styles.append('')
                        return styles
                    # Aplicar o estilo usando apply com axis=1
                    styled_rank_display = df_rank_display.style.apply(highlight_gold_players, axis=1)
                    st.dataframe(styled_rank_display, use_container_width=True)
                        
                    if not df_rank_display.empty:
                        csv = df_rank_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Ranking como CSV",
                            data=csv,
                            file_name='ranking_detalhado_jogadores.csv',
                            mime='text/csv',
                            key='download_ranking_table'
                        )


    # --- Comparação Métrica a Métrica Tab ---
    with tab2:
        # --- SCORE INDEX TAB ---
        st.subheader("📊 ScoreID - Índices de Desempenho (45-100)")

        # Usar o mesmo metrics_by_position_profile definido anteriormente
        # (não redefinir para evitar inconsistências)

        # Filtrar métricas para só as que existem no DataFrame
        for pos, metrics in metrics_by_position_profile.items():
            metrics_by_position_profile[pos] = [m for m in metrics if m in all_metrics_columns]

        profile_selected = st.selectbox(
            "Escolha o Perfil de Posição para calcular o ScoreID",
            list(metrics_by_position_profile.keys())
        )

        # --- Agregação de dados por jogador ---
        metrics_to_aggregate = [m for m in all_metrics_columns if m in final_filtered_df.columns]

        if not final_filtered_df.empty and 'Jogador' in final_filtered_df.columns:
            aggregated_df = final_filtered_df.groupby('Jogador')[metrics_to_aggregate].sum().reset_index()
        else:
            aggregated_df = pd.DataFrame(columns=['Jogador'] + all_metrics_columns)

        player_highlight = st.selectbox(
            "Destacar Jogador na Tabela",
            ['Nenhum'] + aggregated_df['Jogador'].unique().tolist()
        )

        # Usar as métricas já definidas globalmente
        # (offensive_metrics, defensive_metrics, passing_metrics, general_metrics já estão definidas)

        def calculate_aggregated_scores_fut_style(df, metrics_list, min_score=45, max_score=100):
            relevant_metrics = [m for m in metrics_list if m in df.columns]
            if not relevant_metrics:
                return pd.Series([min_score] * len(df), index=df.index)
            df_temp = df[relevant_metrics].copy().fillna(0)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_temp)
            df_scaled = pd.DataFrame(scaled_data, columns=relevant_metrics, index=df.index)
            score_normalized = df_scaled.mean(axis=1).fillna(0)
            return (score_normalized * (max_score - min_score)) + min_score
    
        def get_card_color_and_type(score):
            if score >= 90:
                return "#FFD700", "ICON"
            elif score >= 85:
                return "#FF6B35", "INFORM"
            elif score >= 80:
                return "#8A2BE2", "PURPLE"
            elif score >= 75:
                return "#1E90FF", "RARE"
            else:
                return "#C0C0C0", "COMMON"
    
        if not aggregated_df.empty:
            df_scores_combined = aggregated_df[['Jogador']].copy()
    
            overall_metrics = metrics_by_position_profile[profile_selected]
            overall_metrics = [m for m in overall_metrics if m in all_metrics_columns]
    
            if overall_metrics:
                df_scores_combined['ScoreID'] = calculate_aggregated_scores_fut_style(
                    aggregated_df, overall_metrics
                )
            else:
                df_scores_combined['ScoreID'] = 45.0
    
            df_scores_combined['Score Ofensivo'] = calculate_aggregated_scores_fut_style(
                aggregated_df, offensive_metrics
            )
            df_scores_combined['Score Defensivo'] = calculate_aggregated_scores_fut_style(
                aggregated_df, defensive_metrics
            )
            df_scores_combined['Score de Passe'] = calculate_aggregated_scores_fut_style(
                aggregated_df, passing_metrics
            )
    
            for col in ['ScoreID', 'Score Ofensivo', 'Score Defensivo', 'Score de Passe']:
                if col in df_scores_combined.columns:
                    df_scores_combined[col] = df_scores_combined[col].round(0).astype(int)
    
            df_scores_combined['Cor_Carta'] = df_scores_combined['ScoreID'].apply(lambda x: get_card_color_and_type(x)[0])
            df_scores_combined['Tipo_Carta'] = df_scores_combined['ScoreID'].apply(lambda x: get_card_color_and_type(x)[1])
    
            def highlight_player(row):
                if row['Jogador'] == player_highlight:
                    return ['background-color: #add8e6'] * len(row)
                return [''] * len(row)
    
            display_df = df_scores_combined.drop(['Cor_Carta'], axis=1)
            styled_df = display_df.sort_values(by='ScoreID', ascending=False).style.apply(highlight_player, axis=1)
    
            st.dataframe(styled_df, use_container_width=True)
    
         # Adicionar esta seção após a tabela principal do ScoreID

        # --- TABELA SCOREID POR POSIÇÃO ---
        st.subheader("🎯 ScoreID por Posição - Matriz de Desempenho")
        
        # Filtro para selecionar tipos de score
        score_types_selected = st.multiselect(
            "Selecione os tipos de Score para exibir nas colunas:",
            options=['ScoreID', 'Score Ofensivo', 'Score Defensivo', 'Score de Passe'],
            default=['ScoreID'],
            help="Escolha quais métricas de desempenho deseja visualizar para cada posição"
        )
        
        if not score_types_selected:
            st.warning("⚠️ Selecione pelo menos um tipo de score para exibir a tabela.")
        elif not aggregated_df.empty:
            # Criar DataFrame para armazenar ScoreID por posição para cada jogador
            position_scores_df = pd.DataFrame()
            position_scores_df['Jogador'] = aggregated_df['Jogador']
            
            # Calcular scores para cada posição e tipo selecionado
            for position, metrics in metrics_by_position_profile.items():
                # Filtrar métricas que existem no DataFrame
                position_metrics = [m for m in metrics if m in aggregated_df.columns]
                
                # Remover emojis do nome da posição
                position_name = position.split(' ')[1] if ' ' in position else position
                
                if position_metrics:
                    # Calcular cada tipo de score selecionado
                    for score_type in score_types_selected:
                        if score_type == 'ScoreID':
                            # Usa todas as métricas da posição
                            score = calculate_aggregated_scores_fut_style(
                                aggregated_df, position_metrics
                            ).round(0).astype(int)
                            column_name = f"{position_name}"
                            
                        elif score_type == 'Score Ofensivo':
                            # Usa apenas métricas ofensivas que estão na posição
                            position_offensive_metrics = [m for m in position_metrics if m in offensive_metrics]
                            if position_offensive_metrics:
                                score = calculate_aggregated_scores_fut_style(
                                    aggregated_df, position_offensive_metrics
                                ).round(0).astype(int)
                            else:
                                score = pd.Series([45] * len(aggregated_df))
                            column_name = f"{position_name}_OFF"
                            
                        elif score_type == 'Score Defensivo':
                            # Usa apenas métricas defensivas que estão na posição
                            position_defensive_metrics = [m for m in position_metrics if m in defensive_metrics]
                            if position_defensive_metrics:
                                score = calculate_aggregated_scores_fut_style(
                                    aggregated_df, position_defensive_metrics
                                ).round(0).astype(int)
                            else:
                                score = pd.Series([45] * len(aggregated_df))
                            column_name = f"{position_name}_DEF"
                            
                        elif score_type == 'Score de Passe':
                            # Usa apenas métricas de passe que estão na posição
                            position_passing_metrics = [m for m in position_metrics if m in passing_metrics]
                            if position_passing_metrics:
                                score = calculate_aggregated_scores_fut_style(
                                    aggregated_df, position_passing_metrics
                                ).round(0).astype(int)
                            else:
                                score = pd.Series([45] * len(aggregated_df))
                            column_name = f"{position_name}_PAS"
                        
                        position_scores_df[column_name] = score
                else:
                    # Se não há métricas, preencher com score mínimo para cada tipo selecionado
                    for score_type in score_types_selected:
                        if score_type == 'ScoreID':
                            column_name = f"{position_name}"
                        elif score_type == 'Score Ofensivo':
                            column_name = f"{position_name}_OFF"
                        elif score_type == 'Score Defensivo':
                            column_name = f"{position_name}_DEF"
                        elif score_type == 'Score de Passe':
                            column_name = f"{position_name}_PAS"
                        
                        position_scores_df[column_name] = pd.Series([45] * len(aggregated_df))
            
            # Função para aplicar cores baseada no score
            def color_scoreid_cells(val):
                try:
                    score = float(val)
                    color, _ = get_card_color_and_type(score)
                    return f'background-color: {color}; color: black; font-weight: bold; text-align: center'
                except:
                    return ''
            
            # Aplicar formatação de cores apenas às colunas de score (excluindo 'Jogador')
            score_columns = [col for col in position_scores_df.columns if col != 'Jogador']
            
            # Criar styled dataframe
            styled_position_df = position_scores_df.style.applymap(
                color_scoreid_cells, 
                subset=score_columns
            ).format(
                {col: '{:.0f}' for col in score_columns}
            )
            
            # Destacar jogador selecionado (se houver)
            if player_highlight != 'Nenhum':
                def highlight_selected_player(row):
                    if row['Jogador'] == player_highlight:
                        return ['background-color: #add8e6; border: 2px solid #1E90FF'] * len(row)
                    return [''] * len(row)
                
                styled_position_df = styled_position_df.apply(highlight_selected_player, axis=1)
            
            # Exibir tabela
            st.dataframe(styled_position_df, use_container_width=True)
            
            # --- VISUALIZAÇÃO CARTAS FUT BASEADAS NA POSIÇÃO SELECIONADA ---
            st.subheader("🃏 Cartas FUT - Perfil de Posição Selecionado")
            
            # Calcular scores específicos para a posição selecionada (do filtro principal)
            selected_position_metrics = metrics_by_position_profile[profile_selected]
            selected_position_metrics = [m for m in selected_position_metrics if m in aggregated_df.columns]
            
            if selected_position_metrics and not aggregated_df.empty:
                # Criar DataFrame para as cartas com scores específicos da posição
                cards_df = aggregated_df[['Jogador']].copy()
                
                # ScoreID da posição selecionada
                cards_df['ScoreID'] = calculate_aggregated_scores_fut_style(
                    aggregated_df, selected_position_metrics
                ).round(0).astype(int)
                
                # Scores específicos baseados nas métricas da posição selecionada
                position_offensive_metrics = [m for m in selected_position_metrics if m in offensive_metrics]
                position_defensive_metrics = [m for m in selected_position_metrics if m in defensive_metrics]
                position_passing_metrics = [m for m in selected_position_metrics if m in passing_metrics]
                
                cards_df['Score Ofensivo'] = calculate_aggregated_scores_fut_style(
                    aggregated_df, position_offensive_metrics if position_offensive_metrics else ['Jogador']  # fallback
                ).round(0).astype(int) if position_offensive_metrics else 45
                
                cards_df['Score Defensivo'] = calculate_aggregated_scores_fut_style(
                    aggregated_df, position_defensive_metrics if position_defensive_metrics else ['Jogador']  # fallback
                ).round(0).astype(int) if position_defensive_metrics else 45
                
                cards_df['Score de Passe'] = calculate_aggregated_scores_fut_style(
                    aggregated_df, position_passing_metrics if position_passing_metrics else ['Jogador']  # fallback
                ).round(0).astype(int) if position_passing_metrics else 45
                
                # Adicionar cores das cartas
                cards_df['Cor_Carta'] = cards_df['ScoreID'].apply(lambda x: get_card_color_and_type(x)[0])
                
                # Top 10 jogadores para esta posição
                top_players = cards_df.nlargest(10, 'ScoreID')
                
                # Mostrar métricas consideradas para esta posição
                with st.expander(f"📋 Métricas consideradas para {profile_selected}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**⚔️ Ofensivas:**")
                        if position_offensive_metrics:
                            for metric in position_offensive_metrics:
                                st.write(f"• {metric}")
                        else:
                            st.write("• Nenhuma métrica ofensiva")
                    
                    with col2:
                        st.write("**🛡️ Defensivas:**")
                        if position_defensive_metrics:
                            for metric in position_defensive_metrics:
                                st.write(f"• {metric}")
                        else:
                            st.write("• Nenhuma métrica defensiva")
                    
                    with col3:
                        st.write("**🎯 Passe:**")
                        if position_passing_metrics:
                            for metric in position_passing_metrics:
                                st.write(f"• {metric}")
                        else:
                            st.write("• Nenhuma métrica de passe")
                
                # Funções para as cartas (mesmo código original)
                def get_card_emoji(score):
                    if score >= 90:
                        return "🥇"
                    elif score >= 85:                        
                        return "🔥"
                    elif score >= 80:
                        return "💜"
                    elif score >= 75:
                        return "💎"
                    else:
                        return "⚪"
            
                def get_card_type_name(score):
                    if score >= 90:
                        return "ICON"
                    elif score >= 85:
                        return "INFORM"
                    elif score >= 80:
                        return "PURPLE"
                    elif score >= 75:                        
                        return "RARE"
                    else:
                        return "COMMON"
            
                def get_player_photo_base64(player_name):
                    base_url = "https://raw.githubusercontent.com/mlsaraiva88/football/main/fotos_jogadores"
                    for ext in ["png", "jpg", "jpeg"]:
                        photo_url = f"{base_url}/{player_name}.{ext}"
                        try:
                            response = requests.get(photo_url)
                            if response.status_code == 200:
                                image_data_b64 = base64.b64encode(response.content).decode("ascii")
                                return f"data:image/{ext};base64,{image_data_b64}"
                        except:
                            pass
                    return None
            
                # Renderizar cartas FUT
                if not top_players.empty:
                    num_cards = len(top_players)
                    for i in range(0, num_cards, 5):
                        cols = st.columns(5)
                        batch = top_players.iloc[i:i+5]
                        for idx, (_, player) in enumerate(batch.iterrows()):
                            if idx < len(cols):
                                with cols[idx]:
                                    score = int(player['ScoreID'])
                                    card_emoji = get_card_emoji(score)
                                    card_type = get_card_type_name(score)
                                    card_color = player['Cor_Carta']
                                    player_photo = get_player_photo_base64(player['Jogador'])
            
                                    html_card = f"""
                                    <div style="
                                        background: {card_color};
                                        border-radius: 10px;
                                        padding: 15px;
                                        text-align: center;
                                        color: black;
                                        font-family: Arial, sans-serif;
                                        margin: 5px 0;
                                        border: 2px solid #FFD700;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                    ">
                                    <div style="font-size: 30px; font-weight: bold;">{card_emoji} {score}</div>
                                    <div style="font-size: 10px; margin: 5px 0;">{card_type}</div>
                                    <div style="font-size: 12px; font-weight: bold; margin: 10px 0;">
                                            {player['Jogador'][:15]}{'...' if len(player['Jogador']) > 15 else ''}
                                            {f"<img src='{player_photo}' style='width:80px;height:80px;border-radius:50%;margin:5px 0;'>" if player_photo else ""}
                                    </div>
                                    <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.3);">
                                    <div style="display: flex; justify-content: space-around; font-size: 10px;">
                                        <div>ATK<br><b>{int(player['Score Ofensivo'])}</b></div>
                                        <div>PAS<br><b>{int(player['Score de Passe'])}</b></div>
                                        <div>DEF<br><b>{int(player['Score Defensivo'])}</b></div>
                                    </div>
                                    </div>
                                    """
                                    st.markdown(html_card, unsafe_allow_html=True)
            else:
                st.warning("Não há métricas suficientes para gerar as cartas FUT para esta posição.")
            
            # Adicionar legenda das cores
            st.markdown("""
            **Legenda de Cores:**
            - 🥇 **Dourado (90-100)**: ICON
            - 🔥 **Laranja (85-89)**: INFORM  
            - 💜 **Roxo (80-84)**: PURPLE
            - 💎 **Azul (75-79)**: RARE
            - ⚪ **Prata (45-74)**: COMMON
            """)
            
            # Opção para mostrar estatísticas por posição
            with st.expander("📊 Estatísticas Resumo por Posição"):
                st.write("**Estatísticas por Tipo de Score e Posição:**")
                
                position_stats = {}
                for col in score_columns:
                    avg_score = position_scores_df[col].mean()
                    max_score = position_scores_df[col].max()
                    min_score = position_scores_df[col].min()
                    best_player = position_scores_df.loc[position_scores_df[col].idxmax(), 'Jogador']
                    
                    # Determinar o tipo de score baseado no sufixo da coluna
                    if col.endswith('_OFF'):
                        score_type_label = "Ofensivo"
                    elif col.endswith('_DEF'):
                        score_type_label = "Defensivo" 
                    elif col.endswith('_PAS'):
                        score_type_label = "Passe"
                    else:
                        score_type_label = "Geral"
                    
                    position_name = col.replace('_OFF', '').replace('_DEF', '').replace('_PAS', '')
                    display_name = f"{position_name} ({score_type_label})"
                    
                    position_stats[display_name] = {
                        'Média': round(avg_score, 1),
                        'Máximo': int(max_score),
                        'Mínimo': int(min_score),
                        'Melhor Jogador': best_player
                    }
                
                stats_df = pd.DataFrame(position_stats).T
                st.dataframe(stats_df, use_container_width=True)
                
            # Adicionar informação sobre os sufixos das colunas
            if len(score_types_selected) > 1:
                st.info("""
                **Sufixos das Colunas:**
                - **Sem sufixo**: ScoreID Geral da Posição (todas as métricas da posição)
                - **_OFF**: Score Ofensivo (apenas métricas ofensivas presentes na posição)
                - **_DEF**: Score Defensivo (apenas métricas defensivas presentes na posição)
                - **_PAS**: Score de Passe (apenas métricas de passe presentes na posição)
                
                **Nota**: Se uma posição não tem métricas de um tipo específico, o score será 45 (mínimo).
                """)
        
        else:
            st.warning("Não há dados suficientes para gerar a tabela de ScoreID por posição.")


 
            
    # --- NEW TAB: Quadrant Analysis Scatter Plot ---
    def create_advanced_quadrant_plot(data, metric, highlight_player=None):
        """Cria gráfico de quadrantes avançado com recursos otimizados."""

        # Verificar se a métrica existe no DataFrame
        if metric not in data.columns:
            return None, None

        # Processar dados
        df_processed = data.groupby('Jogador').agg({
            'Clube ou Seleção': lambda x: ', '.join(x.unique()),
            'Posição': lambda x: ', '.join(x.astype(str).unique()),
            metric: 'mean'
        }).reset_index()

        # Limpar dados
        df_processed = df_processed.dropna(subset=[metric])
        df_processed[metric] = pd.to_numeric(df_processed[metric], errors='coerce')
        df_processed = df_processed.dropna(subset=[metric])

        if df_processed.empty or df_processed[metric].nunique() <= 1:
            return None, None

        # Calcular ranking
        df_processed['Rank'] = df_processed[metric].rank(ascending=False, method='dense')

        # Calcular medianas
        median_value = df_processed[metric].median()
        median_rank = df_processed['Rank'].median()

        # Atribuir quadrantes
        def assign_quadrant(row):
            value = row[metric]
            rank = row['Rank']
            if value >= median_value and rank <= median_rank:
                return 'Elite (Alto Valor + Bom Ranking)'
            elif value < median_value and rank > median_rank:
                return 'Baixo Desempenho'
            elif value >= median_value and rank > median_rank:
                return 'Alto Valor + Ranking Baixo'
            else:
                return 'Baixo Valor + Bom Ranking'

        df_processed['Quadrante'] = df_processed.apply(assign_quadrant, axis=1)

        # Cores dos quadrantes
        color_map = {
            'Elite (Alto Valor + Bom Ranking)': '#4CAF50',
            'Baixo Desempenho': '#F44336',
            'Alto Valor + Ranking Baixo': '#FF9800',
            'Baixo Valor + Bom Ranking': '#2196F3'
        }

        # Criar figura
        fig = px.scatter(
            df_processed,
            x=metric,
            y='Rank',
            color='Quadrante',
            color_discrete_map=color_map,
            hover_name='Jogador',
            hover_data={
                'Clube ou Seleção': True,
                'Posição': True,
                metric: ':.2f',
                'Rank': ':.0f',
                'Quadrante': False
            },
            title=f'Análise de Quadrantes: {metric}',
            labels={
                metric: f'{metric} ({normalization_option})',
                'Rank': 'Ranking (1 = Melhor)'
            }
        )
        
        # Inverter eixo Y (ranking)
        fig.update_yaxes(autorange="reversed")
        
        # Adicionar linhas de mediana
        fig.add_hline(y=median_rank, line_dash="dash", line_color="gray", 
                      annotation_text="Mediana Ranking")
        fig.add_vline(x=median_value, line_dash="dash", line_color="gray", 
                      annotation_text="Mediana Valor")
        
        # Adicionar regiões coloridas
        val_min = df_processed[metric].min() * 0.95
        val_max = df_processed[metric].max() * 1.05
        rank_min = 0
        rank_max = df_processed['Rank'].max() * 1.05
        
        # Região Elite (verde)
        fig.add_shape(type="rect",
                      x0=median_value, y0=rank_min, x1=val_max, y1=median_rank,
                      fillcolor='rgba(76, 175, 80, 0.15)', layer="below", line_width=0)
        
        # Região Baixo Desempenho (vermelha)
        fig.add_shape(type="rect",
                      x0=val_min, y0=median_rank, x1=median_value, y1=rank_max,
                      fillcolor='rgba(244, 67, 54, 0.15)', layer="below", line_width=0)
        
        # Região Alto Valor + Ranking Baixo (laranja)
        fig.add_shape(type="rect",
                      x0=median_value, y0=median_rank, x1=val_max, y1=rank_max,
                      fillcolor='rgba(255, 152, 0, 0.1)', layer="below", line_width=0)
        
        # Região Baixo Valor + Bom Ranking (azul)
        fig.add_shape(type="rect",
                      x0=val_min, y0=rank_min, x1=median_value, y1=median_rank,
                      fillcolor='rgba(33, 150, 243, 0.1)', layer="below", line_width=0)
        
        # Destacar jogador selecionado
        if highlight_player and highlight_player in df_processed['Jogador'].values:
            player_data = df_processed[df_processed['Jogador'] == highlight_player].iloc[0]
            fig.add_trace(go.Scatter(
                x=[player_data[metric]],
                y=[player_data['Rank']],
                mode='markers+text',
                marker=dict(size=20, color='yellow', symbol='star', 
                           line=dict(width=3, color='red')),
                text=[f"⭐ {highlight_player}"],
                textposition="top center",
                name=f"Destaque: {highlight_player}",
                showlegend=False
            ))
        
        # Configurar layout
        fig.update_layout(
            template="plotly_white",
            height=600,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig, df_processed
    
    def create_aggregated_quadrant_plot(data, metrics, highlight_player=None):
        """Cria análise agregada de múltiplas métricas."""
        
        # Calcular média das métricas selecionadas
        df_agg = data.copy()
        df_agg['Aggregated_Value'] = df_agg[metrics].mean(axis=1)
        
        # Agrupar por jogador
        df_processed = df_agg.groupby('Jogador').agg({
            'Clube ou Seleção': lambda x: ', '.join(x.unique()),
            'Posição': lambda x: ', '.join(x.astype(str).unique()),
            'Aggregated_Value': 'mean'
        }).reset_index()
        
        # Calcular ranking
        df_processed['Aggregated_Rank'] = df_processed['Aggregated_Value'].rank(ascending=False, method='dense')
        
        # Criar o gráfico usando a função existente
        return create_advanced_quadrant_plot_internal(df_processed, 'Aggregated_Value', 'Aggregated_Rank', 
                                                     "Análise Agregada", highlight_player)
    
    def create_advanced_quadrant_plot_internal(df, val_col, rank_col, title, highlight_player=None):
        """Função interna para criar gráficos de quadrantes."""
        
        median_value = df[val_col].median()
        median_rank = df[rank_col].median()
        
        # Atribuir quadrantes
        def assign_quadrant(row):
            value = row[val_col]
            rank = row[rank_col]
            if value >= median_value and rank <= median_rank:
                return 'Elite'
            elif value < median_value and rank > median_rank:
                return 'Baixo Desempenho'
            elif value >= median_value and rank > median_rank:
                return 'Alto Valor + Ranking Baixo'
            else:
                return 'Baixo Valor + Bom Ranking'
        
        df['Quadrante'] = df.apply(assign_quadrant, axis=1)
        
        # Criar scatter plot
        fig = px.scatter(
            df, x=val_col, y=rank_col, color='Quadrante',
            hover_name='Jogador',
            title=title,
            color_discrete_map={
                'Elite': '#4CAF50',
                'Baixo Desempenho': '#F44336',
                'Alto Valor + Ranking Baixo': '#FF9800',
                'Baixo Valor + Bom Ranking': '#2196F3'
            }
        )
        
        fig.update_yaxes(autorange="reversed")
        fig.add_hline(y=median_rank, line_dash="dash", line_color="gray")
        fig.add_vline(x=median_value, line_dash="dash", line_color="gray")
        
        return fig, df
    
    def render_advanced_quadrant_tab():
        """Interface principal da análise de quadrantes."""
        
        st.header("🎯 Análise Avançada de Quadrantes")
        st.info("💡 Sistema otimizado com processamento robusto e visualizações interativas")
        
        if final_filtered_df.empty:
            st.warning("⚠️ Dados insuficientes. Ajuste os filtros na barra lateral.")
            return
        
        # Configurações
        col1, col2 = st.columns(2)
        with col1:
            highlight_player = st.selectbox(
                "🔍 Jogador para Destaque",
                ['Nenhum'] + sorted(final_filtered_df['Jogador'].unique()),
                key='highlight_quadrant_adv'
            )
            highlight_player = None if highlight_player == 'Nenhum' else highlight_player
        
        with col2:
            analysis_type = st.selectbox(
                "📊 Tipo de Análise",
                ['Individual por Métrica', 'Análise Agregada', 'Ambas'],
                key='analysis_type_quadrant'
            )
        
        # Seleção de métricas
        st.subheader("📊 Seleção de Métricas")
        
        # Perfil de posição
        position_profiles = ['Manual (Selecionar Abaixo)'] + sorted(list(metrics_by_position_profile.keys()))
        selected_profile = st.selectbox(
            "🎯 Perfil de Posição",
            position_profiles,
            key='profile_quadrant_adv'
        )

        # Interface de seleção de métricas
        col1, col2, col3, col4 = st.columns(4)
        selected_metrics = []

        # Métricas por categoria
        categories = [
            (col1, offensive_metrics, "⚔️ Ofensivas", 'off_quadrant_adv'),
            (col2, defensive_metrics, "🛡️ Defensivas", 'def_quadrant_adv'),
            (col3, passing_metrics, "🎯 Passe", 'pass_quadrant_adv'),
            (col4, general_metrics, "📊 Gerais", 'gen_quadrant_adv')
        ]

        for col, metrics_list, label, key in categories:
            with col:
                available = [m for m in metrics_list if m in all_metrics_columns]

                # Definir defaults baseado no perfil
                defaults = []
                if selected_profile != 'Manual (Selecionar Abaixo)':
                    profile_metrics = metrics_by_position_profile.get(selected_profile, [])
                    defaults = [m for m in profile_metrics if m in available]

                # Se há métricas padrão do perfil, usá-las; caso contrário, usar todas
                if defaults:
                    selected = st.multiselect(
                        label,
                        ['Todas'] + available,
                        default=defaults,
                        key=key
                    )
                else:
                    selected = st.multiselect(
                        label,
                        ['Todas'] + available,
                        default=['Todas'] if available else [],
                        key=key
                    )

                if 'Todas' in selected:
                    selected_metrics.extend(available)
                else:
                    selected_metrics.extend(selected)

        selected_metrics = list(set(selected_metrics))  # Remove duplicates
        
        if not selected_metrics:
            st.warning("⚠️ Selecione pelo menos uma métrica")
            return
        
        # Mostrar métricas selecionadas
        with st.expander(f"📋 Métricas Selecionadas ({len(selected_metrics)})"):
            st.write(", ".join(selected_metrics))
        
        # Executar análises
        if analysis_type in ['Análise Agregada', 'Ambas'] and len(selected_metrics) > 1:
            st.subheader("📊 Análise Agregada")
            
            with st.spinner("Processando análise agregada..."):
                try:
                    fig_agg, df_agg = create_aggregated_quadrant_plot(
                        final_filtered_df, selected_metrics, highlight_player
                    )
                    
                    # Métricas resumo
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Jogadores", len(df_agg))
                    with col2:
                        elite_count = len(df_agg[df_agg['Quadrante'] == 'Elite'])
                        st.metric("Elite", elite_count)
                    with col3:
                        st.metric("Valor Mediano", f"{df_agg['Aggregated_Value'].median():.2f}")
                    
                    st.plotly_chart(fig_agg, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erro na análise agregada: {e}")
        
        if analysis_type in ['Individual por Métrica', 'Ambas']:
            st.subheader("📈 Análises Individuais")
            
            for metric in selected_metrics:
                if metric in final_filtered_df.columns:
                    st.markdown(f"### {metric}")
                    
                    with st.spinner(f"Analisando {metric}..."):
                        try:
                            fig, df_metric = create_advanced_quadrant_plot(
                                final_filtered_df, metric, highlight_player
                            )
                            
                            if fig is not None:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.metric("Jogadores", len(df_metric))
                                    st.metric("Variação", f"{df_metric[metric].std():.2f}")
                                    
                                    # Top 3
                                    top3 = df_metric.nsmallest(3, 'Rank')['Jogador'].tolist()
                                    st.markdown("**Top 3:**")
                                    for i, player in enumerate(top3, 1):
                                        st.markdown(f"{i}. {player}")
                            else:
                                st.warning(f"Dados insuficientes para {metric}")
                                
                        except Exception as e:
                            st.error(f"Erro na análise de {metric}: {e}")
    
    # --- EXECUTAR NA ABA 3 ---
    with tab3:
        render_advanced_quadrant_tab()

    # --- NEW TAB: Análise por Perfil de Posição ---
    with tab4:
        st.header("Análise de Métricas por Perfil de Posição")
        st.write("Selecione um perfil de posição para ver as métricas mais relevantes para esse papel e comparar jogadores, independentemente da sua posição declarada.")

        if final_filtered_df.empty:
            st.info("Nenhum dado disponível para análise por perfil de posição. Ajuste os filtros na barra lateral.")
        else:
            position_profile_options = sorted(list(metrics_by_position_profile.keys()))
            selected_position_profile_analysis = st.selectbox(
                "Selecione um Perfil de Posição para Análise",
                position_profile_options,
                key='position_profile_analysis_select'
            )

            # Get the pre-defined metrics for the selected position profile
            predefined_metrics_for_profile = metrics_by_position_profile.get(selected_position_profile_analysis, [])
        
            # Display selected metrics and allow adding more
            st.write(f"Métricas pré-definidas para **{selected_position_profile_analysis}**: {', '.join(predefined_metrics_for_profile) if predefined_metrics_for_profile else 'Nenhuma'}")

            # Allow user to select additional metrics (from all available, excluding predefined)
            all_available_metrics_for_addition = [m for m in all_metrics_columns if m not in predefined_metrics_for_profile]
            additional_metrics_for_profile = st.multiselect(
                "Adicionar Métricas Manuais (Opcional)",
                sorted(all_available_metrics_for_addition),
                key='additional_metrics_position_profile_analysis'
            )

            metrics_to_plot_for_profile = list(set(predefined_metrics_for_profile + additional_metrics_for_profile))
            metrics_to_plot_for_profile = [m for m in metrics_to_plot_for_profile if m in final_filtered_df.columns] # Ensure they exist in the filtered DF

            if not metrics_to_plot_for_profile:
                st.info("Nenhuma métrica selecionada ou disponível para este perfil de posição. Por favor, selecione métricas ou ajuste as configurações.")
            else:
                # Select player to highlight (from all players in final_filtered_df)
                all_players_in_filtered_data = sorted(final_filtered_df['Jogador'].unique().tolist())
                highlight_player_profile_analysis = st.selectbox(
                    "Selecione um Jogador para Destacar (de todos os jogadores filtrados)",
                    ['Nenhum'] + all_players_in_filtered_data,
                    key='highlight_player_profile_analysis'
                )

                # Aggregate data for plotting (mean for each player in the *globally filtered* data)
                # This is key: we are looking at ALL filtered players against the PROFILE metrics.
                df_player_avg_profile = final_filtered_df.groupby('Jogador')[metrics_to_plot_for_profile].mean().reset_index()

                # Calculate overall average for all players in the final_filtered_df for context
                overall_averages_all_filtered = final_filtered_df[metrics_to_plot_for_profile].mean()

                # Prepare data for the summarized report table
                summary_data = {'Métrica': [], f'Valor {highlight_player_profile_analysis}': [], 'Média Geral Filtrada': [], 'Diferença (%)': []}

                # Function to generate plot for each metric in the profile
                def generate_profile_metric_plot(df_data, metric, highlight_player, overall_avg_value):
                    if metric not in df_data.columns or df_data.empty:
                        return

                    # Sort by metric value to make it a distribution bar chart
                    plot_data_metric = df_data[['Jogador', metric]].dropna()
                
                    if plot_data_metric.empty:
                        # Removed to avoid excessive messages in columns
                        # st.info(f"Não há dados válidos para a métrica '{metric}' para os jogadores selecionados.") 
                        return

                    # Define colors: highlight player is red, others are default blue
                    colors = ['lightblue'] * len(plot_data_metric)
                    text_colors = ['black'] * len(plot_data_metric) # Default text color
                    if highlight_player and highlight_player in plot_data_metric['Jogador'].values:
                        highlight_idx_loc = plot_data_metric.index.get_loc(plot_data_metric[plot_data_metric['Jogador'] == highlight_player].index[0])
                        colors[highlight_idx_loc] = 'red'
                        text_colors[highlight_idx_loc] = 'red' # Highlight text color too

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=plot_data_metric['Jogador'],
                        y=plot_data_metric[metric],
                        marker_color=colors,
                        name='Valor do Jogador',
                        text=plot_data_metric['Jogador'], # Add player names as labels
                        textposition='outside', # Position text outside the bars
                        textfont=dict(size=9, color=text_colors), # Adjust font size and color
                        # --- FIX: Changed hovertemplate to a single f-string with escaped Plotly variables ---
                        hovertemplate=f"<b>Jogador:</b> %{{x}}<br><b>{metric}:</b> %{{y:.2f}}<extra></extra>"
                    ))

                    # Add overall average line
                    if pd.notna(overall_avg_value):
                        fig.add_shape(
                            type="line",
                            x0=-0.5, x1=len(plot_data_metric['Jogador']) - 0.5,
                            y0=overall_avg_value, y1=overall_avg_value,
                            line=dict(color="blue", width=2, dash="dash"),
                            name="Média Geral Filtrada"
                        )
                        fig.add_annotation(
                            x=len(plot_data_metric['Jogador']) - 0.5,
                            y=overall_avg_value,
                            text=f"Média Geral: {overall_avg_value:.2f}",
                            showarrow=False,
                            yshift=10,
                            xshift=20,
                            font=dict(color="blue")
                        )

                    fig.update_layout(
                        title=f'{metric}', # Shorter title for smaller plots
                        xaxis_title=None, # Remove x-axis title for compactness
                        yaxis_title=None, # Remove y-axis title for compactness
                        showlegend=False,
                        xaxis={'categoryorder':'total descending'}, # Sort bars
                        height=300, # Smaller height
                        margin=dict(l=20, r=20, t=50, b=20) # Adjust margins
                    )
                    return fig

                st.markdown("---")
                st.subheader(f"Distribuição de Métricas para o Perfil: **{selected_position_profile_analysis}**")

                # Get highlighted player's data for sorting metrics
                highlight_player_avg_data = None
                if highlight_player_profile_analysis != 'Nenhum' and highlight_player_profile_analysis in df_player_avg_profile['Jogador'].values:
                    highlight_player_avg_data = df_player_avg_profile[df_player_avg_profile['Jogador'] == highlight_player_profile_analysis].iloc[0]
            
                # Order metrics based on the highlighted player's performance (descending)
                if highlight_player_avg_data is not None:
                    sorted_metrics_to_plot = sorted(metrics_to_plot_for_profile, key=lambda m: highlight_player_avg_data.get(m, 0), reverse=True)
                else:
                    sorted_metrics_to_plot = metrics_to_plot_for_profile # No highlight player, keep original order
            
                # Create columns for the plots
                cols = st.columns(3)
                col_idx = 0
                for metric in sorted_metrics_to_plot:
                    with cols[col_idx]:
                        fig = generate_profile_metric_plot(df_player_avg_profile, metric, highlight_player_profile_analysis, overall_averages_all_filtered.get(metric))
                        if fig: # Only display if a figure was returned (i.e., data was valid)
                            st.plotly_chart(fig, use_container_width=True)
                    col_idx = (col_idx + 1) % 3 # Move to the next column, wrap around

                    # Populate summary data for each metric
                    player_value = highlight_player_avg_data.get(metric, 0) if highlight_player_avg_data is not None else 0
                    avg_value = overall_averages_all_filtered.get(metric, 0)
                
                    diff_percent = 0
                    if avg_value != 0:
                        diff_percent = ((player_value - avg_value) / avg_value) * 100
                    elif player_value > 0: # If avg is 0 but player has value, it's a huge positive difference
                        diff_percent = 100 # Or some large number to indicate positive
                
                    summary_data['Métrica'].append(metric)
                    summary_data[f'Valor {highlight_player_profile_analysis}'].append(f"{player_value:.2f}")
                    summary_data['Média Geral Filtrada'].append(f"{avg_value:.2f}")
                    summary_data['Diferença (%)'].append(f"{diff_percent:.2f}%")

                # --- Summarized Report ---
                if highlight_player_profile_analysis != 'Nenhum':
                    st.markdown("---")
                    st.subheader(f"Relatório Sintetizado do Desempenho de {highlight_player_profile_analysis} no Perfil de {selected_position_profile_analysis}")
                
                    df_summary = pd.DataFrame(summary_data)

                    def color_difference(val):
                        try:
                            num_val = float(val.replace('%', ''))
                            if num_val > 0:
                                return f'<span style="color: green">{val}</span>'
                            elif num_val < 0:
                                return f'<span style="color: red">{val}</span>'
                            else:
                                return val
                        except ValueError:
                            return val

                    # Apply styling to the 'Diferença (%)' column
                    styled_html_summary_table = df_summary.to_html(escape=False, formatters={'Diferença (%)': color_difference})
                    st.markdown(styled_html_summary_table, unsafe_allow_html=True)
                
                    st.markdown(
                        """
                        <style>
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            font-family: Arial, sans-serif;
                        }
                        th, td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #f2f2f2;
                        }
                        </style>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.info("Selecione um jogador para destacar para ver o relatório sintetizado.")

    # --- NEW TAB CONTENT: Análise de Similaridade (Beta) ---
    # --- ANÁLISE DE SIMILARIDADE DE JOGADORES (OTIMIZADA) ---
    
    def calculate_player_similarity(data, selected_player, metrics, similarity_method='cosine'):
        """
        Calcula similaridade entre jogadores baseada nas métricas selecionadas.
        """
        try:
            # Preparar dados: média por jogador
            player_data = data.groupby('Jogador')[metrics].mean().reset_index()
            
            # Verificar se o jogador existe
            if selected_player not in player_data['Jogador'].values:
                st.error(f"Jogador '{selected_player}' não encontrado nos dados.")
                return pd.DataFrame()
            
            # Tratar valores nulos
            metrics_data = player_data[metrics].fillna(0)
            
            # Normalizar dados usando StandardScaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(metrics_data)
            
            # Obter vetor do jogador selecionado
            player_idx = player_data[player_data['Jogador'] == selected_player].index[0]
            selected_vector = normalized_data[player_idx].reshape(1, -1)
            
            # Calcular similaridades
            if similarity_method == 'cosine':
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(selected_vector, normalized_data)[0]
            else:  # euclidean
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(selected_vector, normalized_data)[0]
                similarities = 1 / (1 + distances)  # Normalizar distâncias
            
            # Criar DataFrame de resultados
            results = pd.DataFrame({
                'Jogador': player_data['Jogador'],
                'Similaridade': similarities
            })
            
            # Remover o próprio jogador e ordenar
            results = results[results['Jogador'] != selected_player]
            results = results.sort_values('Similaridade', ascending=False).reset_index(drop=True)
            
            return results
            
        except Exception as e:
            st.error(f"Erro no cálculo de similaridade: {str(e)}")
            return pd.DataFrame()
    
    def get_metrics_by_profile(profile, metrics_config):
        """Retorna métricas por categoria para um perfil específico."""
        if profile == 'Manual (Selecionar Abaixo)':
            return {
                'offensive': [],
                'defensive': [],
                'passing': [],
                'general': []
            }
        
        profile_metrics = metrics_config.get(profile, [])
        return {
            'offensive': [m for m in profile_metrics if m in offensive_metrics],
            'defensive': [m for m in profile_metrics if m in defensive_metrics],
            'passing': [m for m in profile_metrics if m in passing_metrics],
            'general': [m for m in profile_metrics if m in general_metrics]
        }
    
    def validate_metrics_selection(metrics, available_columns):
        """Valida e filtra métricas selecionadas."""
        return [m for m in metrics if m in available_columns]
    
    def render_similarity_analysis_tab():
        """Renderiza a aba de análise de similaridade."""
        
        st.header("🔍 Análise de Similaridade de Jogadores")
        
        # Status da funcionalidade
        st.info("💡 **Funcionalidade Melhorada**: Análise baseada nos dados carregados com múltiplos algoritmos de similaridade.")
        
        # Verificar se há dados disponíveis
        if final_filtered_df.empty:
            st.warning("⚠️ Nenhum dado disponível. Ajuste os filtros na barra lateral.")
            return
        
        # Configurações principais
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            # Seleção do jogador
            all_players = sorted(final_filtered_df['Jogador'].unique())
            selected_player = st.selectbox(
                "🏃‍♂️ Selecione o Jogador de Referência",
                all_players,
                key='similarity_player'
            )
        
        with col_config2:
            # Método de similaridade
            similarity_method = st.selectbox(
                "📊 Método de Similaridade",
                ['cosine', 'euclidean'],
                format_func=lambda x: 'Cosseno' if x == 'cosine' else 'Euclidiana',
                key='similarity_method'
            )
        
        # Seção de seleção de métricas
        st.subheader("📈 Seleção de Métricas")
        
        # Perfil de posição
        position_profiles = ['Manual (Selecionar Abaixo)'] + sorted(list(metrics_by_position_profile.keys()))
        selected_profile = st.selectbox(
            "🎯 Perfil de Posição (Pré-definido)",
            position_profiles,
            key='similarity_profile'
        )
        
        # Obter métricas do perfil
        profile_metrics = get_metrics_by_profile(selected_profile, metrics_by_position_profile)
        
        # Interface de seleção de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        selected_metrics = []
        
        with col1:
            available_offensive = [m for m in offensive_metrics if m in all_metrics_columns]
            default_offensive = profile_metrics['offensive'] if available_offensive else []
            
            selected_off = st.multiselect(
                "⚔️ Métricas Ofensivas",
                ['Todas'] + available_offensive,
                default=default_offensive if default_offensive else (['Todas'] if available_offensive else []),
                key='similarity_offensive'
            )
            
            if 'Todas' in selected_off:
                selected_metrics.extend(available_offensive)
            else:
                selected_metrics.extend(selected_off)
        
        with col2:
            available_defensive = [m for m in defensive_metrics if m in all_metrics_columns]
            default_defensive = profile_metrics['defensive'] if available_defensive else []
            
            selected_def = st.multiselect(
                "🛡️ Métricas Defensivas",
                ['Todas'] + available_defensive,
                default=default_defensive if default_defensive else (['Todas'] if available_defensive else []),
                key='similarity_defensive'
            )
            
            if 'Todas' in selected_def:
                selected_metrics.extend(available_defensive)
            else:
                selected_metrics.extend(selected_def)
        
        with col3:
            available_passing = [m for m in passing_metrics if m in all_metrics_columns]
            default_passing = profile_metrics['passing'] if available_passing else []
            
            selected_pass = st.multiselect(
                "🎯 Métricas de Passe",
                ['Todas'] + available_passing,
                default=default_passing if default_passing else (['Todas'] if available_passing else []),
                key='similarity_passing'
            )
            
            if 'Todas' in selected_pass:
                selected_metrics.extend(available_passing)
            else:
                selected_metrics.extend(selected_pass)
        
        with col4:
            available_general = [m for m in general_metrics if m in all_metrics_columns]
            default_general = profile_metrics['general'] if available_general else []
            
            selected_gen = st.multiselect(
                "📊 Métricas Gerais",
                ['Todas'] + available_general,
                default=default_general if default_general else (['Todas'] if available_general else []),
                key='similarity_general'
            )
            
            if 'Todas' in selected_gen:
                selected_metrics.extend(available_general)
            else:
                selected_metrics.extend(selected_gen)
        
        # Validar métricas selecionadas
        final_metrics = validate_metrics_selection(list(set(selected_metrics)), final_filtered_df.columns.tolist())
        
        if not final_metrics:
            st.warning("⚠️ Selecione pelo menos uma métrica para análise.")
            return
        
        # Mostrar métricas selecionadas
        with st.expander(f"📋 Métricas Selecionadas ({len(final_metrics)})"):
            st.write(", ".join(final_metrics))
        
        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                top_n = st.slider(
                    "Número de jogadores similares",
                    min_value=5,
                    max_value=min(20, len(final_filtered_df['Jogador'].unique()) - 1),
                    value=10,
                    key='similarity_top_n'
                )
            
            with col_adv2:
                min_similarity = st.slider(
                    "Similaridade mínima",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    key='similarity_min_threshold'
                )
        
        # Realizar análise
        if st.button("🚀 Executar Análise", type="primary"):
            with st.spinner("Calculando similaridades..."):
                results = calculate_player_similarity(
                    final_filtered_df,
                    selected_player, 
                    final_metrics, 
                    similarity_method
                )
                
                if results.empty:
                    st.error("❌ Não foi possível calcular similaridades. Verifique os dados.")
                    return
                
                # Filtrar por similaridade mínima
                results = results[results['Similaridade'] >= min_similarity]
                
                if results.empty:
                    st.warning(f"⚠️ Nenhum jogador encontrado com similaridade >= {min_similarity:.2f}")
                    return
                
                # Mostrar resultados
                st.subheader(f"🎯 Jogadores Mais Similares a **{selected_player}**")
                
                # Métricas de resumo
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.metric("Jogadores Analisados", len(results))
                
                with col_summary2:
                    st.metric("Similaridade Máxima", f"{results['Similaridade'].max():.3f}")
                
                with col_summary3:
                    st.metric("Similaridade Média", f"{results['Similaridade'].mean():.3f}")
                
                # Tabela de resultados
                display_results = results.head(top_n).copy()
                display_results['Ranking'] = range(1, len(display_results) + 1)
                display_results['Similaridade (%)'] = (display_results['Similaridade'] * 100).round(1)
                
                # Exibir tabela
                st.dataframe(
                    display_results[['Ranking', 'Jogador', 'Similaridade (%)']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Gráfico de similaridade (se há mais de 3 resultados)
                if len(display_results) >= 3:
                    st.subheader("📊 Visualização da Similaridade")
                    
                    chart_data = display_results.head(10).set_index('Jogador')['Similaridade (%)']
                    st.bar_chart(chart_data, height=400)
                
                # Opção de download
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Resultados CSV",
                    data=csv,
                    file_name=f"similaridade_{selected_player.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
    
    # --- EXECUTAR NA ABA 5 ---
    with tab5:
        render_similarity_analysis_tab()

    
    with tab6:
        # ==============================================================================
        # Início do Código para ser colado dentro do bloco 'with tab6:'
        # ==============================================================================
        
        import streamlit as st
        import json
        
        # CSS customizado para um design moderno e responsivo
        st.markdown("""
        <style>
            /* Importa a fonte Inter do Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            html, body, .stApp {
                font-family: 'Inter', sans-serif;
                background-color: #f0f2f6; /* Fundo mais suave */
                color: #2d3748;
            }
        
            /* Estilo para o cabeçalho principal */
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                text-align: center;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .main-header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
        
            /* Estilo para os cards de posição */
            .position-card {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border: none;
                border-radius: 20px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                border-left: 5px solid #667eea;
            }
            
            .position-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.15);
                border-left-color: #764ba2;
            }
            
            .position-title {
                font-size: 1.4rem;
                font-weight: 700;
                color: #2d3748;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .kpi-section {
                margin: 1rem 0;
            }
            
            .kpi-title {
                font-size: 1rem;
                font-weight: 600;
                color: #667eea;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .kpi-content {
                background: #f7fafc;
                padding: 1rem;
                border-radius: 12px;
                font-size: 0.9rem;
                line-height: 1.6;
                color: #4a5568;
                border-left: 3px solid #e2e8f0;
            }
            
            .kpi-qualitative {
                border-left-color: #48bb78; /* Verde */
            }
            
            .kpi-quantitative {
                border-left-color: #ed8936; /* Laranja */
            }
            
            /* Estilo para os cards de estatísticas */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); /* Responsivo */
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
            }
            
            .stat-number {
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
            }
            
            .stat-label {
                font-size: 1rem;
                opacity: 0.9;
                margin-top: 0.5rem;
            }
            
            /* Estilo para o painel de edição */
            .edit-panel {
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                margin: 2rem 0;
            }
        
            /* Estilo para os botões */
            .stButton>button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none !important; /* Sobrescreve o estilo padrão do Streamlit */
                padding: 0.75rem 2rem;
                border-radius: 25px;
                color: white !important; /* Sobrescreve o estilo padrão do Streamlit */
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
        
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
        
            .position-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .search-container {
                background: white;
                padding: 1rem;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                margin-bottom: 2rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # ==============================================================================
        # Dados e estado da sessão
        # ==============================================================================
        
        # Definição da estrutura de dados categorizada
        categorized_kpi_data = {
            "Posição": [
                "🧤 Goalkeeper",
                "🧱 Centre-back",
                "🛡️ Full-back",
                "🧲 Defensive Midfielder (#6)",
                "🧠 Central Midfielder (#8)",
                "🎯 Attacking Midfielder (#10)",
                "🪂 Winger (Extremo)",
                "🎯 Striker (Avançado)"
            ],
            "KPI Qualitativos": [
                {
                    "Ofensivos": [],
                    "Defensivos": ["Sair da baliza (tempo, decisão)", "Shot-stopping médio/longo alcance", "Comunicação com linha defensiva"],
                    "Passe": ["Envolvimento no build-up (curto/longo)"],
                    "Geral": ["Posicionamento em cruzamentos e bolas paradas"]
                },
                {
                    "Ofensivos": [],
                    "Defensivos": ["Posicionamento com/sem bola (linha alta, coberturas)", "Reação a bolas nas costas", "Consistência em duelos (aéreos/chão)", "Proteção da área e bloqueios"],
                    "Passe": ["Perfil de passe (seguro/progressivo)"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Timing e frequência de corridas ofensivas", "Overlaps/underlaps no apoio ofensivo", "Participação no build-up (progressivo/seguro)"],
                    "Defensivos": ["Reação em transições defensivas", "Cobertura defensiva ao CB/extremo"],
                    "Passe": [],
                    "Geral": []
                },
                {
                    "Ofensivos": [],
                    "Defensivos": ["Comportamento defensivo (antecipa/reage)", "Cobertura a CB/FB em transições"],
                    "Passe": ["Scanning antes de receber", "360º Awareness", "Papel no build-up (receber e progredir)", "Tipologia de passe (seguro/progressivo/vertical)"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Movimento para receber entre linhas", "Suporte ao ataque em 2ª vaga"],
                    "Defensivos": ["Recuperação defensiva em pressing alto"],
                    "Passe": ["Criação de ligações (progressão/último terço)"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Movimentos entre linhas", "Suporte ofensivo (último terço e finalização)", "Pressão e recuperação em transição"],
                    "Defensivos": [],
                    "Passe": ["Criação de chances (key passes, through balls)", "Decisão sob pressão (drible/passe)"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Perfil (inverter para dentro vs largura por fora)", "Frequência/direção de 1v1s", "Movimentos sem bola (2º poste, diagonais)", "Impacto em espaços abertos vs curtos"],
                    "Defensivos": ["Trabalho defensivo (recuperações, pressão)"],
                    "Passe": [],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Timing de desmarcações em profundidade", "Finalização (1 toque, compostura)", "Presença em bolas paradas", "Envolvimento na pressão alta"],
                    "Defensivos": [],
                    "Passe": ["Jogo de costas e ligação"],
                    "Geral": []
                }
            ],
            "KPI Quantitativos": [
                {
                    "Ofensivos": [],
                    "Defensivos": ["Defesas Totais", "% Defesas", "xG Evitado", "Saídas a Cruzamentos Ganhas"],
                    "Passe": ["Passes Curtos Certos", "Passes Longos Certos"],
                    "Geral": []
                },
                {
                    "Ofensivos": [],
                    "Defensivos": ["Duelos Defensivos Ganhos", "Duelos Aéreos Ganhos", "Intercepções", "Alívios", "Carrinhos ganhas"],
                    "Passe": ["Passes Totais Certos", "Passes Progressivos Certos"],
                    "Geral": ["Perdas Totais"]
                },
                {
                    "Ofensivos": ["Corridas Seguidas", "Cruzamentos Certos", "Assistências", "Assistências para Remate", "Dribles Certos", "Toques na Área"],
                    "Defensivos": ["Duelos Defensivos Ganhos", "Intercepções"],
                    "Passe": ["Passes Progressivos"],
                    "Geral": []
                },
                {
                    "Ofensivos": [],
                    "Defensivos": ["Recuperações", "Intercepções", "Duelos Defensivos Ganhos", "Duelos de Bola Livre Ganhos"],
                    "Passe": ["Passes Totais Certos", "Passes Progressivos Certos", "Passes Verticais Certos", "Assistências para Remate"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Remates à Baliza", "Golos", "Duelos Ofensivos Ganhos"],
                    "Defensivos": ["Recuperação no Meio Campo Adversário"],
                    "Passe": ["Passes Progressivos Certos", "Passes Último Terço", "Assistências", "Segundas Assistências"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Dribles Certos", "Remates Totais", "Remates à Baliza", "xA", "xG", "Toques na Área", "Recuperações Altas", "Duelos Ofensivos Ganhos"],
                    "Defensivos": [],
                    "Passe": ["Assistências", "Assistências para Remate", "Passes Progressivos Certos"],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Dribles Certos", "Cruzamentos Certos", "Assistências", "Assistências para Remate", "Golos", "Remates à Baliza", "xA", "xG", "Corridas Seguidas", "Toques na Área", "Duelos Ofensivos Ganhos", "Recuperações Altas"],
                    "Defensivos": [],
                    "Passe": [],
                    "Geral": []
                },
                {
                    "Ofensivos": ["Golos", "xG", "Remates Totais", "Remates à Baliza", "Toques na Área", "Recuperações Altas", "Duelos Ofensivos Ganhos"],
                    "Defensivos": ["Duelos Aéreos Ganhos"],
                    "Passe": ["Assistências", "Assistências para Remate", "Passes Progressivos"],
                    "Geral": []
                }
            ]
        }
        
        # Verificação e inicialização dos dados na sessão
        if 'kpi_data' not in st.session_state:
            st.session_state.kpi_data = categorized_kpi_data.copy()
            
        # Se o formato de dados for o antigo (lista de strings), converte para o novo
        if st.session_state.kpi_data["KPI Qualitativos"] and isinstance(st.session_state.kpi_data["KPI Qualitativos"][0], str):
            st.session_state.kpi_data = categorized_kpi_data.copy()
        
        # ==============================================================================
        # Funções de conversão para lidar com a nova estrutura de dados
        # ==============================================================================
        
        def string_to_dict(kpi_string, kpi_type):
            """Converte a string do text_area de volta para o dicionário de KPIs."""
            kpi_dict = {"Ofensivos": [], "Defensivos": [], "Passe": [], "Geral": []}
            current_category = None
            lines = kpi_string.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('---') and line.endswith('---'):
                    category = line.replace('---', '').strip()
                    if category in kpi_dict:
                        current_category = category
                    else:
                        current_category = None  # Ignora categorias inválidas
                elif current_category:
                    kpi_dict[current_category].append(line)
                    
            return kpi_dict
        
        # ==============================================================================
        # Layout da Aplicação Streamlit
        # ==============================================================================
        
        # Header principal
        st.markdown("""
        <div class="main-header">
            <h1>⚽ Football KPI Dashboard</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Análise Avançada de Performance por Posição</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cards de estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">8</div>
                <div class="stat-label">Posições</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);">
                <div class="stat-number">50+</div>
                <div class="stat-label">KPIs Qualitativos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); box-shadow: 0 8px 25px rgba(168, 237, 234, 0.3);">
                <div class="stat-number">60+</div>
                <div class="stat-label">KPIs Quantitativos</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stat-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); box-shadow: 0 8px 25px rgba(252, 182, 159, 0.3);">
                <div class="stat-number">110+</div>
                <div class="stat-label">Total KPIs</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Abas internas para dashboard e editor, agora dentro do tab6
        tab_dashboard, tab_editor = st.tabs(["🎯 **Dashboard KPI**", "✏️ **Editor Avançado**"])
        
        with tab_dashboard:
            st.markdown("### 🏟️ Análise por Posição")
            
            # Campo de pesquisa/filtro
            search_term = st.text_input("🔍 Pesquisar posição...", placeholder="Digite o nome da posição")
            
            # Filtra as posições com base no termo de pesquisa
            filtered_positions = []
            for i, position in enumerate(st.session_state.kpi_data["Posição"]):
                if not search_term or search_term.lower() in position.lower():
                    filtered_positions.append(i)
            
            # Exibe os cards de posição
            for i in filtered_positions:
                position_name = st.session_state.kpi_data["Posição"][i]
                qualitative_kpis = st.session_state.kpi_data["KPI Qualitativos"][i]
                quantitative_kpis = st.session_state.kpi_data["KPI Quantitativos"][i]
                
                # Início do card de posição com HTML
                st.markdown(f"""
                <div class="position-card">
                    <div class="position-title">{position_name}</div>
                """, unsafe_allow_html=True)
                
                # Cria as duas colunas para os KPIs
                col_qual, col_quant = st.columns(2)
        
                with col_qual:
                    st.markdown(f"""
                        <div class="kpi-section">
                            <div class="kpi-title">📋 KPIs Qualitativos</div>
                            <div class="kpi-content kpi-qualitative">
                    """, unsafe_allow_html=True)
                    for category, kpi_list in qualitative_kpis.items():
                        if kpi_list:
                            st.markdown(f"**{category}:**")
                            st.markdown("\n".join([f"- {item.strip()}" for item in kpi_list]))
                    st.markdown("""
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
                with col_quant:
                    st.markdown(f"""
                        <div class="kpi-section">
                            <div class="kpi-title">📊 KPIs Quantitativos</div>
                            <div class="kpi-content kpi-quantitative">
                    """, unsafe_allow_html=True)
                    for category, kpi_list in quantitative_kpis.items():
                        if kpi_list:
                            st.markdown(f"**{category}:**")
                            st.markdown("\n".join([f"- {item.strip()}" for item in kpi_list]))
                    st.markdown("""
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
                # Fim do card, com a div de fecho
                st.markdown("</div>", unsafe_allow_html=True)
        
        
        with tab_editor:
            st.markdown("### ✏️ Editor de KPIs")
            
            # Painel de seleção
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <div class="edit-panel">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">🎯 Selecionar Posição</h4>
                </div>
                """, unsafe_allow_html=True)
                
                selected_position = st.selectbox(
                    "Escolha a posição:",
                    options=st.session_state.kpi_data["Posição"],
                    label_visibility="collapsed"
                )
                
                position_index = st.session_state.kpi_data["Posição"].index(selected_position)
                
                # Painel de ações
                st.markdown("""
                <div class="edit-panel">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">⚡ Ações Rápidas</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Resetar Tudo", use_container_width=True):
                    st.session_state.kpi_data = categorized_kpi_data.copy()
                    st.success("✅ Dados resetados!")
                    st.rerun()
                
                # Botão de download de JSON
                json_data = json.dumps(st.session_state.kpi_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Exportar JSON",
                    data=json_data,
                    file_name="kpis_futebol.json",
                    mime="application/json",
                    use_container_width=True
                )
        
            with col2:
                st.markdown(f"""
                <div class="edit-panel">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">Editando: {selected_position}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Campos de edição para KPIs Qualitativos
                st.markdown("**📋 KPIs Qualitativos:**")
                
                qual_data = st.session_state.kpi_data["KPI Qualitativos"][position_index]
                edited_qualitative_ofensivos = st.text_area("Ofensivos", value="\n".join(qual_data.get("Ofensivos", [])), height=100, help="Separe os KPIs com uma quebra de linha (enter)")
                edited_qualitative_defensivos = st.text_area("Defensivos", value="\n".join(qual_data.get("Defensivos", [])), height=100, help="Separe os KPIs com uma quebra de linha (enter)")
                edited_qualitative_passe = st.text_area("Passe", value="\n".join(qual_data.get("Passe", [])), height=100, help="Separe os KPIs com uma quebra de linha (enter)")
                edited_qualitative_geral = st.text_area("Geral", value="\n".join(qual_data.get("Geral", [])), height=100, help="Separe os KPIs com uma quebra de linha (enter)")
        
                # Campos de edição para KPIs Quantitativos
                st.markdown("**📊 KPIs Quantitativos:**")
                quant_data = st.session_state.kpi_data["KPI Quantitativos"][position_index]
                edited_quantitative_ofensivos = st.text_area("Ofensivos", value="\n".join(quant_data.get("Ofensivos", [])), height=100, key="quant_ofensivos", help="Separe os KPIs com uma quebra de linha (enter)")
                edited_quantitative_defensivos = st.text_area("Defensivos", value="\n".join(quant_data.get("Defensivos", [])), height=100, key="quant_defensivos", help="Separe os KPIs com uma quebra de linha (enter)")
                edited_quantitative_passe = st.text_area("Passe", value="\n".join(quant_data.get("Passe", [])), height=100, key="quant_passe", help="Separe os KPIs com uma quebra de linha (enter)")
                edited_quantitative_geral = st.text_area("Geral", value="\n".join(quant_data.get("Geral", [])), height=100, key="quant_geral", help="Separe os KPIs com uma quebra de linha (enter)")
        
                # Botões de salvar e pré-visualizar
                col_save, col_preview = st.columns([1, 1])
                
                with col_save:
                    if st.button("💾 Salvar Alterações", type="primary", use_container_width=True):
                        # Atualiza a estrutura de dados com os novos valores
                        st.session_state.kpi_data["KPI Qualitativos"][position_index]["Ofensivos"] = [item.strip() for item in edited_qualitative_ofensivos.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Qualitativos"][position_index]["Defensivos"] = [item.strip() for item in edited_qualitative_defensivos.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Qualitativos"][position_index]["Passe"] = [item.strip() for item in edited_qualitative_passe.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Qualitativos"][position_index]["Geral"] = [item.strip() for item in edited_qualitative_geral.split('\n') if item.strip()]
        
                        st.session_state.kpi_data["KPI Quantitativos"][position_index]["Ofensivos"] = [item.strip() for item in edited_quantitative_ofensivos.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Quantitativos"][position_index]["Defensivos"] = [item.strip() for item in edited_quantitative_defensivos.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Quantitativos"][position_index]["Passe"] = [item.strip() for item in edited_quantitative_passe.split('\n') if item.strip()]
                        st.session_state.kpi_data["KPI Quantitativos"][position_index]["Geral"] = [item.strip() for item in edited_quantitative_geral.split('\n') if item.strip()]
                        
                        st.success(f"✅ {selected_position} atualizado!")
                        st.rerun()
                
                with col_preview:
                    if st.button("👁️ Preview", use_container_width=True):
                        st.markdown("### 🔍 Preview das Alterações")
                        
                        # Renderiza o preview com as novas categorias
                        preview_qualitative = {
                            "Ofensivos": [item.strip() for item in edited_qualitative_ofensivos.split('\n') if item.strip()],
                            "Defensivos": [item.strip() for item in edited_qualitative_defensivos.split('\n') if item.strip()],
                            "Passe": [item.strip() for item in edited_qualitative_passe.split('\n') if item.strip()],
                            "Geral": [item.strip() for item in edited_qualitative_geral.split('\n') if item.strip()]
                        }
                        
                        preview_quantitative = {
                            "Ofensivos": [item.strip() for item in edited_quantitative_ofensivos.split('\n') if item.strip()],
                            "Defensivos": [item.strip() for item in edited_quantitative_defensivos.split('\n') if item.strip()],
                            "Passe": [item.strip() for item in edited_quantitative_passe.split('\n') if item.strip()],
                            "Geral": [item.strip() for item in edited_quantitative_geral.split('\n') if item.strip()]
                        }
                        
                        def render_kpis(kpi_dict):
                            html_content = ""
                            for category, kpi_list in kpi_dict.items():
                                if kpi_list:
                                    html_content += f"**{category}:**\n"
                                    html_content += "\n".join([f"- {item.strip()}" for item in kpi_list])
                                    html_content += "\n"
                            return html_content
        
                        st.markdown(f"""
                        <div class="position-card">
                            <div class="position-title">{selected_position}</div>
                            
                            <div class="kpi-section">
                                <div class="kpi-title">📋 KPIs Qualitativos</div>
                                <div class="kpi-content kpi-qualitative">
                                    {render_kpis(preview_qualitative)}
                                </div>
                            </div>
                            
                            <div class="kpi-section">
                                <div class="kpi-title">📊 KPIs Quantitativos</div>
                                <div class="kpi-content kpi-quantitative">
                                    {render_kpis(preview_quantitative)}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Footer da aplicação
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #718096;">
            <p style="margin: 0; font-size: 0.9rem;">
                ⚽ <strong>Scouting</strong> | Inteligência Desportiva
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;">
                Versão 3.0 | Design by Miguel Saraiva
            </p>
        </div>
        """, unsafe_allow_html=True)
        # ==============================================================================
        # Fim do Código para ser colado dentro do bloco 'with tab6:'
        # ==============================================================================
    # --- TAB 7: ARQUÉTIPOS DE JOGADORES (CLUSTERING OTIMIZADO) ---
    with tab7:
        st.header("🎭 Análise de Arquétipos Posicionais")
        st.info("💡 Clustering inteligente de jogadores com ponderação de métricas e descrição automática de arquétipos")
        
        # Imports necessários
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        # ============================================================================
        # FUNÇÕES AUXILIARES
        # ============================================================================
        
        def categorize_metric(metric_name):
            """Categoriza uma métrica com base no seu nome."""
            offensive_keywords = ['Golos', 'Assistências', 'Remates', 'Dribles', 'Cruzamentos', 'Toques na Área', 'xG', 'xA']
            defensive_keywords = ['Desarmes', 'Intercepções', 'Bloqueios', 'Carrinhos', 'Alívios', 'Duelos Defensivos', 'Recuperações']
            passing_keywords = ['Passes', 'Progressivos', 'Chave', 'Longos', 'profundidade', 'terço final']
            
            metric_lower = metric_name.lower()
            
            if any(kw.lower() in metric_lower for kw in offensive_keywords):
                return 'Ofensivo'
            elif any(kw.lower() in metric_lower for kw in defensive_keywords):
                return 'Defensivo'
            elif any(kw.lower() in metric_lower for kw in passing_keywords):
                return 'Construção'
            else:
                return 'Geral'
        
        def generate_archetype_description(cluster_avg, overall_avg, profile_metrics, threshold=0.20):
            """
            Gera descrição automática do arquétipo baseada nas métricas mais proeminentes.
            
            Args:
                cluster_avg: Média das métricas do cluster
                overall_avg: Média geral das métricas
                profile_metrics: Lista de métricas analisadas
                threshold: Limite para considerar uma métrica proeminente (20% acima da média)
            
            Returns:
                tuple: (nome, descrição, ícone, cor)
            """
            # Calcular diferença percentual
            overall_avg_safe = overall_avg.replace(0, 1e-6)
            diff_pct = (cluster_avg - overall_avg) / overall_avg_safe
            
            # Filtrar métricas proeminentes
            prominent_metrics = diff_pct[diff_pct > threshold].sort_values(ascending=False)
            
            # Categorizar métricas proeminentes
            categories = {'Ofensivo': 0, 'Defensivo': 0, 'Construção': 0, 'Geral': 0}
            
            for metric in prominent_metrics.index:
                category = categorize_metric(metric)
                categories[category] += 1
            
            # Determinar categoria dominante
            dominant_categories = [cat for cat, count in categories.items() if count > 0]
            dominant_categories.sort(key=lambda x: categories[x], reverse=True)
            
            # Gerar nome do arquétipo
            if len(dominant_categories) >= 3:
                archetype_name = "Jogador Completo"
                icon = "⭐"
                color = "#FFD700"
            elif len(dominant_categories) == 2:
                archetype_name = f"{dominant_categories[0]} / {dominant_categories[1]}"
                icon = "✨"
                color = "#FF6B35"
            elif len(dominant_categories) == 1:
                archetype_name = f"{dominant_categories[0]}"
                icon = "🎯"
                color_map = {
                    'Ofensivo': '#E91E63',
                    'Defensivo': '#2196F3',
                    'Construção': '#4CAF50',
                    'Geral': '#607D8B'
                }
                color = color_map.get(dominant_categories[0], '#607D8B')
            else:
                archetype_name = "Perfil Neutro"
                icon = "⚪"
                color = "#9E9E9E"
            
            # Gerar descrição
            if prominent_metrics.empty:
                top_metrics = cluster_avg.nlargest(3)
                metrics_text = ", ".join([f"**{m}** ({cluster_avg[m]:.2f})" for m in top_metrics.index])
                description = f"Perfil equilibrado sem destaque significativo. Principais métricas: {metrics_text}."
            else:
                metrics_list = []
                for metric in prominent_metrics.head(5).index:
                    value = cluster_avg[metric]
                    diff = prominent_metrics[metric] * 100
                    metrics_list.append(f"**{metric}** ({value:.2f}, +{diff:.0f}%)")
                
                metrics_text = ", ".join(metrics_list)
                description = f"Destaca-se em métricas de **{archetype_name}**. Principais forças: {metrics_text}."
            
            return archetype_name, description, icon, color
        
        def prepare_clustering_data(df, metrics, min_games):
            """Prepara os dados para clustering."""
            # Filtrar jogadores com mínimo de jogos
            player_counts = df['Jogador'].value_counts()
            valid_players = player_counts[player_counts >= min_games].index
            
            df_filtered = df[df['Jogador'].isin(valid_players)]
            
            if df_filtered.empty:
                return None, None
            
            # Agregar por jogador
            df_agg = df_filtered.groupby('Jogador')[metrics].mean().reset_index()
            
            # Limpar dados
            df_agg = df_agg.fillna(0)
            df_agg = df_agg.replace([np.inf, -np.inf], 0)
            
            return df_agg, df_agg['Jogador'].values
        
        def apply_weights_and_scale(X, weights):
            """Aplica pesos e normalização aos dados."""
            # Normalizar primeiro
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Aplicar pesos
            weight_array = np.array(list(weights.values()))
            X_weighted = X_scaled * weight_array
            
            return X_weighted, scaler
        
        # ============================================================================
        # INTERFACE PRINCIPAL
        # ============================================================================
        
        if final_filtered_df.empty:
            st.warning("⚠️ Nenhum dado disponível. Ajuste os filtros na barra lateral.")
            st.stop()
        
        # Seleção do perfil posicional
        st.subheader("🎯 Configuração da Análise")
        
        col_pos, col_clusters = st.columns([2, 1])
        
        with col_pos:
            position_profile = st.selectbox(
                "📍 Perfil Posicional",
                list(metrics_by_position_profile.keys()),
                key='clustering_position',
                help="Escolha a posição para análise específica de arquétipos"
            )
        
        with col_clusters:
            n_clusters = st.slider(
                "🎲 Número de Arquétipos",
                min_value=2,
                max_value=8,
                value=4,
                help="Quantidade de grupos distintos a identificar"
            )
        
        # Obter métricas do perfil
        profile_metrics = metrics_by_position_profile.get(position_profile, [])
        profile_metrics = [m for m in profile_metrics if m in final_filtered_df.columns]
        
        if not profile_metrics:
            st.error("❌ Nenhuma métrica disponível para este perfil.")
            st.stop()
        
        # Configurações adicionais
        with st.expander("⚙️ Configurações Avançadas", expanded=False):
            col_config1, col_config2 = st.columns(2)
            
            with col_config1:
                min_games = st.number_input(
                    "Mínimo de registros por jogador",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="Filtrar jogadores com poucos dados"
                )
            
            with col_config2:
                prominence_threshold = st.slider(
                    "Sensibilidade de destaque (%)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    help="Diferença mínima para considerar métrica proeminente"
                ) / 100
        
        # Ponderação de métricas
        st.subheader("⚖️ Ponderação de Métricas")
        
        with st.expander("🎚️ Ajustar Pesos das Métricas", expanded=False):
            st.info("💡 Defina a importância de cada métrica. Peso 1.0 = padrão, >1.0 = mais importante, <1.0 = menos importante")
            
            # Remover métricas duplicadas para evitar erros de chaves duplicadas
            unique_profile_metrics = list(dict.fromkeys(profile_metrics))

            # Organizar em colunas
            weights = {}
            num_cols = 3
            cols = st.columns(num_cols)

            for i, metric in enumerate(unique_profile_metrics):
                with cols[i % num_cols]:
                    # Usar índice i na key para garantir unicidade
                    weights[metric] = st.slider(
                        metric,
                        min_value=0.0,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key=f"weight_{metric}_{position_profile}_{i}"
                    )
            
            # Botão para resetar pesos
            if st.button("🔄 Resetar Todos os Pesos para 1.0"):
                st.rerun()
        
        # Mostrar resumo da configuração
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
            <strong>📊 Configuração Atual:</strong><br>
            • Posição: {position_profile}<br>
            • Métricas: {len(profile_metrics)}<br>
            • Arquétipos: {n_clusters}<br>
            • Min. jogos: {min_games}
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================================
        # EXECUTAR CLUSTERING
        # ============================================================================
        
        if st.button("🚀 Executar Análise de Arquétipos", type="primary", use_container_width=True):
            with st.spinner("🔄 Processando clustering..."):
                try:
                    # 1. Preparar dados
                    df_agg, player_names = prepare_clustering_data(
                        final_filtered_df, 
                        profile_metrics, 
                        min_games
                    )
                    
                    if df_agg is None:
                        st.error("❌ Nenhum jogador atende ao critério mínimo de jogos.")
                        st.stop()
                    
                    # 2. Aplicar pesos e normalização
                    X = df_agg[profile_metrics].values
                    X_weighted, scaler = apply_weights_and_scale(X, weights)
                    
                    # 3. Executar K-Means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
                    clusters = kmeans.fit_predict(X_weighted)
                    
                    # 4. PCA para visualização
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_weighted)
                    
                    # 5. Adicionar resultados ao dataframe
                    df_agg['Cluster'] = clusters
                    df_agg['PC1'] = X_pca[:, 0]
                    df_agg['PC2'] = X_pca[:, 1]
                    
                    # 6. Calcular médias
                    overall_avg = df_agg[profile_metrics].mean()
                    
                    # ====================================================================
                    # VISUALIZAÇÕES E RESULTADOS
                    # ====================================================================
                    
                    st.success("✅ Clustering concluído com sucesso!")
                    
                    # Estatísticas gerais
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("👥 Jogadores", len(df_agg))
                    
                    with col2:
                        st.metric("🎭 Arquétipos", n_clusters)
                    
                    with col3:
                        st.metric("📊 Métricas", len(profile_metrics))
                    
                    with col4:
                        variance = sum(pca.explained_variance_ratio_) * 100
                        st.metric("📈 Variância (PCA)", f"{variance:.1f}%")
                    
                    # Gráfico PCA
                    st.subheader("🗺️ Mapa de Arquétipos")
                    
                    # Gerar informações dos arquétipos
                    archetype_info = {}
                    for cluster_id in range(n_clusters):
                        cluster_data = df_agg[df_agg['Cluster'] == cluster_id]
                        cluster_avg = cluster_data[profile_metrics].mean()
                        
                        name, desc, icon, color = generate_archetype_description(
                            cluster_avg, overall_avg, profile_metrics, prominence_threshold
                        )
                        
                        archetype_info[cluster_id] = {
                            'name': name,
                            'description': desc,
                            'icon': icon,
                            'color': color,
                            'count': len(cluster_data)
                        }
                    
                    # Adicionar nome do arquétipo ao dataframe
                    df_agg['Arquétipo'] = df_agg['Cluster'].apply(
                        lambda x: f"{archetype_info[x]['icon']} {archetype_info[x]['name']}"
                    )
                    
                    # Criar scatter plot
                    fig_pca = px.scatter(
                        df_agg,
                        x='PC1',
                        y='PC2',
                        color='Arquétipo',
                        hover_name='Jogador',
                        hover_data={
                            'PC1': ':.2f',
                            'PC2': ':.2f',
                            'Cluster': True
                        },
                        title=f"Distribuição de Arquétipos - {position_profile}",
                        labels={
                            'PC1': f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                            'PC2': f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                        },
                        height=600
                    )
                    
                    # Customizar visualização
                    fig_pca.update_traces(
                        marker=dict(size=10, line=dict(width=1.5, color='white')),
                        textposition='top center'
                    )
                    
                    # Adicionar centróides
                    centroids_pca = pca.transform(kmeans.cluster_centers_)
                    for cluster_id in range(n_clusters):
                        fig_pca.add_trace(go.Scatter(
                            x=[centroids_pca[cluster_id, 0]],
                            y=[centroids_pca[cluster_id, 1]],
                            mode='markers+text',
                            marker=dict(
                                symbol='x',
                                size=20,
                                color=archetype_info[cluster_id]['color'],
                                line=dict(width=3, color='white')
                            ),
                            text=[f"C{cluster_id+1}"],
                            textposition="top center",
                            name=f"Centróide {cluster_id+1}",
                            showlegend=False
                        ))
                    
                    fig_pca.update_layout(
                        template="plotly_white",
                        hovermode='closest',
                        legend=dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Análise detalhada por arquétipo
                    st.markdown("---")
                    st.subheader("🔍 Caracterização dos Arquétipos")
                    
                    for cluster_id in range(n_clusters):
                        info = archetype_info[cluster_id]
                        cluster_data = df_agg[df_agg['Cluster'] == cluster_id]
                        cluster_avg = cluster_data[profile_metrics].mean()
                        
                        # Card do arquétipo
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {info['color']}20 0%, {info['color']}05 100%);
                            border-left: 6px solid {info['color']};
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin: 1rem 0;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        ">
                            <h3 style="color: {info['color']}; margin: 0 0 0.5rem 0; font-size: 1.5rem;">
                                {info['icon']} {info['name']} (Cluster {cluster_id+1})
                            </h3>
                            <p style="color: #555; margin: 0.5rem 0; font-size: 1rem; line-height: 1.6;">
                                {info['description']}
                            </p>
                            <p style="margin: 1rem 0 0 0; font-weight: 600; color: {info['color']};">
                                👥 {info['count']} jogadores ({info['count']/len(df_agg)*100:.1f}% do total)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detalhes do cluster
                        col_players, col_metrics = st.columns([1, 1])
                        
                        with col_players:
                            st.markdown("**🌟 Jogadores Representativos:**")
                            top_players = cluster_data.nlargest(8, profile_metrics[0])['Jogador'].tolist()
                            for idx, player in enumerate(top_players, 1):
                                st.write(f"{idx}. {player}")
                        
                        with col_metrics:
                            st.markdown("**📊 Top 5 Métricas:**")
                            top_metrics = cluster_avg.nlargest(5)
                            for metric, value in top_metrics.items():
                                overall_val = overall_avg[metric]
                                diff = ((value - overall_val) / (overall_val + 1e-6)) * 100
                                color = "🟢" if diff > 0 else "🔴"
                                st.write(f"{color} **{metric}**: {value:.2f} ({diff:+.0f}%)")
                        
                        # Radar chart
                        if len(profile_metrics) >= 3:
                            with st.expander(f"📈 Perfil Completo - {info['name']}"):
                                fig_radar = go.Figure()
                                
                                # Média do cluster
                                metrics_closed = profile_metrics + [profile_metrics[0]]
                                values_closed = cluster_avg.tolist() + [cluster_avg.tolist()[0]]
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=values_closed,
                                    theta=metrics_closed,
                                    fill='toself',
                                    name=info['name'],
                                    line_color=info['color'],
                                    fillcolor=f"rgba{tuple(list(int(info['color'][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}"
                                ))
                                
                                # Média geral
                                overall_closed = overall_avg.tolist() + [overall_avg.tolist()[0]]
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=overall_closed,
                                    theta=metrics_closed,
                                    fill='none',
                                    name='Média Geral',
                                    line=dict(color='gray', dash='dot', width=2)
                                ))
                                
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, max(cluster_avg.max(), overall_avg.max()) * 1.2]
                                        )
                                    ),
                                    showlegend=True,
                                    title=f"Perfil Radar - {info['name']}",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Tabela completa
                    st.markdown("---")
                    st.subheader("📋 Classificação Completa de Jogadores")
                    
                    # Preparar tabela
                    display_df = df_agg[['Jogador', 'Arquétipo', 'Cluster']].copy()
                    display_df = display_df.sort_values(['Cluster', 'Jogador'])
                    
                    # Aplicar cores
                    def highlight_cluster(row):
                        cluster_id = row['Cluster']
                        color = archetype_info[cluster_id]['color']
                        return [f'background-color: {color}15; border-left: 4px solid {color}'] * len(row)
                    
                    styled_table = display_df.style.apply(highlight_cluster, axis=1)
                    
                    st.dataframe(
                        styled_table,
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download
                    col_download1, col_download2 = st.columns(2)
                    
                    with col_download1:
                        csv_data = df_agg.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Análise Completa (CSV)",
                            data=csv_data,
                            file_name=f"arquetipos_{position_profile.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_download2:
                        # Criar relatório resumido
                        summary_data = []
                        for cluster_id in range(n_clusters):
                            info = archetype_info[cluster_id]
                            summary_data.append({
                                'Cluster': cluster_id + 1,
                                'Arquétipo': info['name'],
                                'Jogadores': info['count'],
                                'Percentagem': f"{info['count']/len(df_agg)*100:.1f}%"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_csv = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Download Resumo (CSV)",
                            data=summary_csv,
                            file_name=f"resumo_arquetipos_{position_profile.replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise: {str(e)}")
                    with st.expander("🔍 Detalhes do Erro"):
                        st.exception(e)
# ---------------------------------------------------------------------------------------------------------------------------------
    # --- BLOCO INTRODUTÓRIO: VISÃO GERAL DO JOGADOR SELECIONADO ---
    with tab8:
        st.header("📊 Visão Geral e Ranking de Jogadores")
        
        # ==============================
        # FUNÇÃO PARA EXTRAIR CLUBE DA COLUNA 'JOGO'
        # ==============================
        def extract_club_from_game(jogo_str):
            """Extrai o clube mais provável baseado na coluna 'Jogo' (clubes separados por '-')"""
            if pd.isna(jogo_str) or not isinstance(jogo_str, str):
                return "N/A"
            
            # Separar por '-' e limpar espaços
            clubs = [club.strip() for club in jogo_str.split('-') if club.strip()]
            if not clubs:
                return "N/A"
            
            # Se há apenas um clube, retornar ele
            if len(clubs) == 1:
                return clubs[0]
            
            # Se há dois clubes, retornar o primeiro (assumindo que é o clube do jogador)
            return clubs[0]
        
        # ==============================
        # FUNÇÃO PARA CALCULAR MÉTRICAS DETALHADAS
        # ==============================
        def calculate_detailed_metrics(player_data, recent_data):
            """Calcula métricas detalhadas por categoria"""
            
            # Função auxiliar para cálculo seguro
            def safe_calc(func, *args):
                try:
                    return func(*args)
                except:
                    return 0
            
            metrics = {}
            
            # Métricas Ofensivas
            metrics['ofensiva'] = {
                'total_golos': safe_calc(lambda: player_data['Golos'].sum()),
                'media_golos': safe_calc(lambda: player_data['Golos'].mean()),
                'total_assistencias': safe_calc(lambda: player_data['Assistências'].sum()),
                'media_assistencias': safe_calc(lambda: player_data['Assistências'].mean()),
                'total_xg': safe_calc(lambda: player_data['xG'].sum()),
                'media_xg': safe_calc(lambda: player_data['xG'].mean()),
                'total_xa': safe_calc(lambda: player_data['xA'].sum()),
                'media_xa': safe_calc(lambda: player_data['xA'].mean()),
                'ultimo_golo': safe_calc(lambda: recent_data['Golos'].iloc[0]) if not recent_data.empty else 0,
                'ultima_assistencia': safe_calc(lambda: recent_data['Assistências'].iloc[0]) if not recent_data.empty else 0
            }
            
            # Métricas Defensivas
            metrics['defensiva'] = {
                'total_intercecoes': safe_calc(lambda: player_data['Intercepções'].sum()) if 'Intercepções' in player_data.columns else 0,
                'media_intercecoes': safe_calc(lambda: player_data['Intercepções'].mean()) if 'Intercepções' in player_data.columns else 0,
                'total_recuperacoes': safe_calc(lambda: player_data['Recuperações Totais'].sum()) if 'Recuperações Totais' in player_data.columns else 0,
                'media_recuperacoes': safe_calc(lambda: player_data['Recuperações Totais'].mean()) if 'Recuperações Totais' in player_data.columns else 0,
                'total_duelos_def': safe_calc(lambda: player_data['Duelos Defensivos Ganhos'].sum()) if 'Duelos Defensivos Ganhos' in player_data.columns else 0,
                'media_duelos_def': safe_calc(lambda: player_data['Duelos Defensivos Ganhos'].mean()) if 'Duelos Defensivos Ganhos' in player_data.columns else 0
            }
            
            # Métricas de Passe
            metrics['passe'] = {
                'total_passes': safe_calc(lambda: player_data['Passes Totais Certos'].sum()) if 'Passes Totais Certos' in player_data.columns else 0,
                'media_passes': safe_calc(lambda: player_data['Passes Totais Certos'].mean()) if 'Passes Totais Certos' in player_data.columns else 0,
                'total_passesseg': safe_calc(lambda: player_data['Passes Progressivos Certos'].sum()) if 'Passes Progressivos Certos' in player_data.columns else 0,
                'media_passesseg': safe_calc(lambda: player_data['Passes Progressivos Certos'].mean()) if 'Passes Progressivos Certos' in player_data.columns else 0
            }
            
            # Métricas Gerais
            metrics['geral'] = {
                'total_minutos': safe_calc(lambda: player_data['Minutos'].sum()),
                'media_minutos': safe_calc(lambda: player_data['Minutos'].mean()),
                'total_accoes': safe_calc(lambda: player_data['Ações Totais'].sum()) if 'Ações Totais' in player_data.columns else 0,
                'media_accoes': safe_calc(lambda: player_data['Ações Totais'].mean()) if 'Ações Totais' in player_data.columns else 0,
                'total_acao_sucesso': safe_calc(lambda: player_data['Ações Sucesso'].sum()) if 'Ações Sucesso' in player_data.columns else 0,
                'media_acao_sucesso': safe_calc(lambda: player_data['Ações Sucesso'].mean()) if 'Ações Sucesso' in player_data.columns else 0
            }
            
            # Disciplina
            metrics['disciplina'] = {
                'total_amarelos': safe_calc(lambda: player_data['Cartões Amarelos'].sum()) if 'Cartões Amarelos' in player_data.columns else 0,
                'media_amarelos': safe_calc(lambda: player_data['Cartões Amarelos'].mean()) if 'Cartões Amarelos' in player_data.columns else 0,
                'total_vermelhos': safe_calc(lambda: player_data['Cartões Vermelhos'].sum()) if 'Cartões Vermelhos' in player_data.columns else 0,
                'media_vermelhos': safe_calc(lambda: player_data['Cartões Vermelhos'].mean()) if 'Cartões Vermelhos' in player_data.columns else 0
            }
            
            return metrics
        
        # ==============================
        # SELEÇÃO DE JOGADOR
        # ==============================
        if df.empty:
            st.error("❌ Dados não carregados. Por favor, carregue um ficheiro Excel primeiro.")
            st.stop()
        
        jogadores_disponiveis = df['Jogador'].dropna().unique().tolist() if 'Jogador' in df.columns else []
        
        if not jogadores_disponiveis:
            st.error("⚠️ Nenhum jogador encontrado nos dados carregados.")
            st.stop()
        
        # Usar o multiselect do sidebar se disponível, ou selectbox normal
        if 'jogador_multiselect' in st.session_state and st.session_state.jogador_multiselect:
            # Se há jogadores selecionados no multiselect, usar o primeiro
            if 'Todos' not in st.session_state.jogador_multiselect:
                selected_player = st.selectbox("👤 Seleciona o jogador", sorted(jogadores_disponiveis), key="player_select")
            else:
                selected_player = st.selectbox("👤 Seleciona o jogador", sorted(jogadores_disponiveis), key="player_select")
        else:
            selected_player = st.selectbox("👤 Seleciona o jogador", sorted(jogadores_disponiveis), key="player_select")
        
        # ==============================
        # DEFINIR COLUNA DE DATA
        # ==============================
        possible_date_cols = ['Date', 'Jogo_Data', 'Data', 'Data_Jogo', 'Data do Jogo', 'Match_Date', 'GameDate']
        col_data_jogo = next((c for c in possible_date_cols if c in df.columns), None)
        
        # ==============================
        # ANÁLISE DO JOGADOR SELECIONADO
        # ==============================
        player_df = df[df['Jogador'] == selected_player].copy()
        
        if col_data_jogo:
            player_df = player_df.sort_values(col_data_jogo, ascending=False)
            player_df_recent = player_df.head(1)
            player_df_for_avg = player_df.tail(-1)  # Todos exceto o último
        else:
            st.warning("⚠️ Nenhuma coluna de data encontrada. Usando último registo disponível.")
            player_df_recent = player_df.tail(1)
            player_df_for_avg = player_df.head(-1)
        
        if player_df.empty or player_df_recent.empty:
            st.warning("❗ Dados insuficientes para este jogador.")
            st.stop()
        
        # Extrair clube do último jogo
        ultimo_jogo = player_df_recent.iloc[0]
        clube_do_jogador = "N/A"
        if 'Jogo' in ultimo_jogo.index:
            clube_do_jogador = extract_club_from_game(ultimo_jogo['Jogo'])
        elif 'Clube ou Seleção' in ultimo_jogo.index:
            clube_do_jogador = ultimo_jogo['Clube ou Seleção']
        
        # Posições
        posicao_mais_jogada = player_df['Posição'].mode().iloc[0] if 'Posição' in player_df.columns and not player_df['Posição'].empty else "N/A"
        posicao_ultimo_jogo = player_df_recent['Posição'].iloc[0] if 'Posição' in player_df_recent.columns else "N/A"
        
        # Calcular métricas detalhadas
        player_metrics = calculate_detailed_metrics(player_df, player_df_recent)
        
        # ==============================
        # VISUALIZAÇÃO PRINCIPAL - VISÃO GERAL DO JOGADOR
        # ==============================
        st.subheader(f"👤 {selected_player} - Visão Geral")
        
        # Cabeçalho com informações do jogador
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            color: white;
            margin-bottom: 20px;
            text-align: center;
        ">
            <h2 style="margin: 0; font-size: 2em;">⚽ {selected_player}</h2>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                🏟️ <strong>Clube:</strong> {clube_do_jogador} | 
                📍 <strong>Posição Típica:</strong> {posicao_mais_jogada} | 
                🎯 <strong>Posição Último Jogo:</strong> {posicao_ultimo_jogo}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # ==============================
        # COMPARAÇÃO: TOTAL vs ÚLTIMO JOGO
        # ==============================
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Estatísticas Totais (Temporada)")
            total_col = st.container()
            
            # Métricas principais em cards
            metrics_data = [
                ("⏱️", "Minutos", f"{player_metrics['geral']['total_minutos']:.0f}", "Jogados"),
                ("⚽", "Golos", f"{player_metrics['ofensiva']['total_golos']:.0f}", "Marcados"),
                ("🅰️", "Assistências", f"{player_metrics['ofensiva']['total_assistencias']:.0f}", "Ofertas"),
                ("🎯", "xG", f"{player_metrics['ofensiva']['total_xg']:.2f}", "Esperados"),
                ("🎪", "xA", f"{player_metrics['ofensiva']['total_xa']:.2f}", "Esperados"),
                ("⚡", "Ações", f"{player_metrics['geral']['total_accoes']:.0f}", "Realizadas")
            ]
            
            for i, (emoji, titulo, valor, sub) in enumerate(metrics_data):
                if i % 2 == 0:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 8px 0;
                        color: white;
                        text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 1.8em; margin-bottom: 5px;">{emoji}</div>
                        <div style="font-size: 1.4em; font-weight: bold; margin-bottom: 2px;">{valor}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">{titulo} - {sub}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #4ECDC4 0%, #FF6B6B 100%);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 8px 0;
                        color: white;
                        text-align: center;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 1.8em; margin-bottom: 5px;">{emoji}</div>
                        <div style="font-size: 1.4em; font-weight: bold; margin-bottom: 2px;">{valor}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">{titulo} - {sub}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Selecionar o Position Profile para análise
            available_profiles = list(metrics_by_position_profile.keys())
            if not available_profiles:
                st.warning("⚠️ Nenhum Position Profile disponível nos dados.")
                st.stop()
            
            selected_profile = st.selectbox(
                "🎯 **Position Profile para Análise:**",
                available_profiles,
                key="position_profile_select"
            )
            
            # Obter as métricas deste profile
            profile_metrics = metrics_by_position_profile[selected_profile]
            
            # Dados do último jogo
            data_ultimo = ultimo_jogo[col_data_jogo] if col_data_jogo else "Data não disponível"
            st.markdown(f"### 🏆 Último Jogo ({data_ultimo})")
            recent_col = st.container()
            
            ultimo_jogo_info = f"📅 **Jogo:** {ultimo_jogo.get('Jogo', 'N/A')}\n\n"
            ultimo_jogo_info += f"📅 **Data:** {data_ultimo}\n\n"
            ultimo_jogo_info += f"""
            ⏱️ **Minutos:** {player_metrics['geral']['media_minutos']:.0f}
            ⚽ **Golos:** {player_metrics['ofensiva']['ultimo_golo']:.0f}
            🅰️ **Assistências:** {player_metrics['ofensiva']['ultima_assistencia']:.0f}
            🎯 **xG:** {player_df_recent['xG'].iloc[0] if 'xG' in player_df_recent.columns else 0:.2f}
            📍 **Posição:** {posicao_ultimo_jogo}
            """
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 20px;
                color: white;
                box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            ">
                {ultimo_jogo_info.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar os 4 jogos anteriores
            st.markdown("---")
            st.markdown("#### 📋 Últimos 5 Jogos")
            
            if len(player_df) >= 5:
                ultimos_5_jogos = player_df.head(5)
            else:
                ultimos_5_jogos = player_df
            
            for idx, (_, jogo) in enumerate(ultimos_5_jogos.iterrows()):
                if col_data_jogo and col_data_jogo in jogo.index:
                    data_jogo = jogo[col_data_jogo]
                else:
                    data_jogo = f"Jogo {idx + 1}"
                
                nome_jogo = jogo.get('Jogo', 'Jogo não disponível')
                minutos = jogo.get('Minutos', 0)
                golos = jogo.get('Golos', 0)
                assistencias = jogo.get('Assistências', 0)
                posicao = jogo.get('Posição', 'N/A')
                
                # Definir cor baseada na posição
                if idx == 0:
                    cor_jogo = "#FF6B6B"  # Vermelho para o mais recente
                    emoji = "🏆"
                else:
                    cor_jogo = "#4ECDC4"  # Ciano para os anteriores
                    emoji = "⚽"
                
                st.markdown(f"""
                <div style="
                    background: {cor_jogo}20;
                    border-left: 4px solid {cor_jogo};
                    border-radius: 8px;
                    padding: 12px;
                    margin: 5px 0;
                ">
                    <div style="font-weight: bold; color: {cor_jogo}; margin-bottom: 5px;">
                        {emoji} {data_jogo} - {nome_jogo}
                    </div>
                    <div style="font-size: 0.9em; color: #666;">
                        ⏱️ {minutos:.0f} min | ⚽ {golos:.0f} | 🅰️ {assistencias:.0f} | 📍 {posicao}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ==============================
        # MÉTRICAS POR CATEGORIA COM MÉDIAS E POSITION PROFILE
        # ==============================
        st.markdown("---")
        st.subheader(f"📈 Análise por Categorias - {selected_profile}")
        
        # Destacar métricas do Position Profile selecionado
        profile_metrics_available = [m for m in profile_metrics if m in player_df.columns]
        
        if profile_metrics_available:
            st.info(f"🎯 **Métricas Prioritárias para {selected_profile}:** {', '.join(profile_metrics_available[:5])}{'...' if len(profile_metrics_available) > 5 else ''}")
        
        # 1. CATEGORIA INFO GERAL
        st.markdown("#### 📋 **Info Geral**")
        info_geral_cols = st.columns(2)
        
        with info_geral_cols[0]:
            st.markdown("**📊 Estatísticas Principais**")
            # Métricas básicas: Golos, Assistências, xG, xA
            info_gerais_stats = {
                'Golos': player_metrics['ofensiva'],
                'Assistências': player_metrics['ofensiva'],
                'xG': player_metrics['ofensiva'],
                'xA': player_metrics['ofensiva']
            }
            
            st.markdown(f"""
            <div style="background: #E8F4FD20; border-left: 4px solid #2196F3; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #2196F3;">Média por Jogo:</div>
                ⚽ <strong>Golos:</strong> {player_metrics['ofensiva']['media_golos']:.2f}<br>
                🅰️ <strong>Assistências:</strong> {player_metrics['ofensiva']['media_assistencias']:.2f}<br>
                🎯 <strong>xG:</strong> {player_metrics['ofensiva']['media_xg']:.2f}<br>
                🎪 <strong>xA:</strong> {player_metrics['ofensiva']['media_xa']:.2f}<br>
                <div style="font-weight: bold; margin-top: 10px; margin-bottom: 5px; color: #2196F3;">Totais:</div>
                ⚽ <strong>Golos:</strong> {player_metrics['ofensiva']['total_golos']:.0f}<br>
                🅰️ <strong>Assistências:</strong> {player_metrics['ofensiva']['total_assistencias']:.0f}
            </div>
            """, unsafe_allow_html=True)
        
        with info_geral_cols[1]:
            st.markdown("**⏱️ Atividade Geral**")
            st.markdown(f"""
            <div style="background: #E8F4FD20; border-left: 4px solid #2196F3; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #2196F3;">Média por Jogo:</div>
                ⏱️ <strong>Minutos:</strong> {player_metrics['geral']['media_minutos']:.1f}<br>
                ⚡ <strong>Ações:</strong> {player_metrics['geral']['media_accoes']:.1f}<br>
                ✅ <strong>Sucesso:</strong> {player_metrics['geral']['media_acao_sucesso']:.1f}<br>
                <div style="font-weight: bold; margin-top: 10px; margin-bottom: 5px; color: #2196F3;">Totais:</div>
                ⏱️ <strong>Minutos:</strong> {player_metrics['geral']['total_minutos']:.0f}<br>
                ⚡ <strong>Ações:</strong> {player_metrics['geral']['total_accoes']:.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # 2. CATEGORIAS COM MÉTRICAS DO METRIC_CATEGORY_MAP
        categorias = [
            ("⚽", "Ofensiva", "#FF6B6B"),
            ("🛡️", "Defensiva", "#4ECDC4"),
            ("🎯", "Passe", "#45B7D1"),
            ("📊", "Geral", "#96CEB4")
        ]
        
        cat_cols = st.columns(4)
        
        for i, (emoji, nome, cor) in enumerate(categorias):
            with cat_cols[i]:
                # Ajustar nome para exibição
                nome_exibicao = nome + "s" if nome != "Passe" else nome
                st.markdown(f"#### {emoji} {nome_exibicao}")
                
                # Filtrar métricas desta categoria que estão no perfil selecionado
                category_profile_metrics = [m for m in profile_metrics_available if m in final_metric_category_map and final_metric_category_map[m] == nome]
                
                if category_profile_metrics:
                    # Construir HTML de forma mais simples e segura
                    html_parts = []
                    html_parts.append(f'<div style="background: {cor}20; border-left: 4px solid {cor}; padding: 15px; border-radius: 8px;">')
                    html_parts.append(f'<div style="font-weight: bold; margin-bottom: 10px; color: {cor};">🌟 Métricas do {selected_profile}:</div>')
                    
                    for metric in category_profile_metrics[:4]:  # Mostrar até 4 métricas principais
                        if metric in player_df.columns:
                            # Calcular média e total para esta métrica
                            metric_avg = player_df[metric].mean()
                            metric_total = player_df[metric].sum()
                            
                            # Determinar emoji baseado na métrica
                            metric_emoji = "⚡"  # padrão
                            if 'Golos' in metric:
                                metric_emoji = "⚽"
                            elif 'Assist' in metric:
                                metric_emoji = "🅰️"
                            elif 'Interc' in metric:
                                metric_emoji = "🔄"
                            elif 'Pass' in metric:
                                metric_emoji = "📋"
                            elif 'Duel' in metric:
                                metric_emoji = "⚔️"
                            elif 'Remat' in metric:
                                metric_emoji = "🎯"
                            
                            html_metric = f'''<div style="margin: 8px 0; padding: 5px; background: {cor}10; border-radius: 5px;">
{metric_emoji} <strong>{metric}:</strong><br>
<span style="font-size: 0.9em;">📊 Média: {metric_avg:.2f}</span> | <span style="font-size: 0.9em;">📈 Total: {metric_total:.0f}</span>
</div>'''
                            html_parts.append(html_metric)
                    
                    html_parts.append('</div>')
                    final_html = '\n'.join(html_parts)
                    st.markdown(final_html, unsafe_allow_html=True)
                else:
                    # Mostrar que não há métricas específicas do perfil nesta categoria
                    st.markdown(f"""
                    <div style="background: {cor}20; border-left: 4px solid {cor}; padding: 15px; border-radius: 8px; text-align: center; color: #666;">
                        <em>Sem métricas prioritárias do {selected_profile} em {nome_exibicao}</em>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ==============================
        # MELHOR JOGO DO JOGADOR
        # ==============================
        st.markdown("---")
        
        # Identificar o melhor jogo (baseado em múltiplos fatores)
        if not player_df.empty:
            # Calcular score composto para cada jogo
            def calculate_game_score(row):
                score = 0
                if 'Golos' in row.index:
                    score += row['Golos'] * 10
                if 'Assistências' in row.index:
                    score += row['Assistências'] * 8
                if 'xG' in row.index:
                    score += row['xG'] * 5
                if 'Ações Sucesso' in row.index:
                    score += row['Ações Sucesso'] * 0.1
                return score
            
            player_df['game_score'] = player_df.apply(calculate_game_score, axis=1)
            melhor_jogo = player_df.loc[player_df['game_score'].idxmax()]
            score_melhor = melhor_jogo['game_score']
            
            # Obter data do melhor jogo
            if col_data_jogo and col_data_jogo in melhor_jogo.index:
                data_melhor_jogo = melhor_jogo[col_data_jogo]
                st.subheader(f"🏆 Melhor Jogo da Temporada ({data_melhor_jogo})")
            else:
                data_melhor_jogo = "Data não disponível"
                st.subheader("🏆 Melhor Jogo da Temporada")
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                border-radius: 15px;
                padding: 20px;
                color: #333;
                text-align: center;
                box-shadow: 0 6px 12px rgba(255, 215, 0, 0.3);
            ">
                <h3 style="margin: 0 0 10px 0; font-size: 1.5em;">🏆 Melhor Jogo Identificado</h3>
                <p style="margin: 0; font-size: 1.2em; font-weight: bold;">⚽ {melhor_jogo.get('Jogo', 'N/A')}</p>
                <p style="margin: 5px 0 0 0; font-size: 1.1em;">Score de Performance: {score_melhor:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalhes do melhor jogo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**📊 Estatísticas do Jogo:**")
                st.write(f"⚽ Golos: {melhor_jogo.get('Golos', 0):.0f}")
                st.write(f"🅰️ Assistências: {melhor_jogo.get('Assistências', 0):.0f}")
                st.write(f"🎯 xG: {melhor_jogo.get('xG', 0):.2f}")
            with col2:
                st.markdown("**📈 Métricas Avançadas:**")
                st.write(f"🎪 xA: {melhor_jogo.get('xA', 0):.2f}")
                st.write(f"⚡ Ações: {melhor_jogo.get('Ações Totais', 0):.0f}")
                st.write(f"✅ Sucesso: {melhor_jogo.get('Ações Sucesso', 0):.0f}")
            with col3:
                st.markdown("**📍 Contexto:**")
                st.write(f"📍 Posição: {melhor_jogo.get('Posição', 'N/A')}")
                st.write(f"⏱️ Minutos: {melhor_jogo.get('Minutos', 0):.0f}")
                if col_data_jogo:
                    st.write(f"📅 Data: {melhor_jogo[col_data_jogo]}")
        
        # ==============================
        # RANKING DE JOGADORES
        # ==============================
        st.markdown("---")
        st.subheader(f"📊 Ranking Global de Jogadores - {selected_profile}")
        st.info(f"🎯 **Ranking baseado nas melhores performances individuais - Métricas prioritárias para {selected_profile}**")
        
        # Calcular ranking para todos os jogadores
        all_players_ranking = []
        
        for player in jogadores_disponiveis:
            try:
                player_data = df[df['Jogador'] == player].copy()
                if len(player_data) < 2:  # Skip players with too few games
                    continue
                
                # Ordenar por data se disponível
                if col_data_jogo:
                    player_data = player_data.sort_values(col_data_jogo, ascending=False)
                
                recent_game = player_data.iloc[0]
                historical_data = player_data.iloc[1:] if len(player_data) > 1 else player_data
                
                if historical_data.empty:
                    continue
                
                # Calcular melhorias por categoria usando métricas do Position Profile
                ofensiva_improvement = 0
                defensiva_improvement = 0
                passe_improvement = 0
                geral_improvement = 0
                
                # Métricas relevantes para o Position Profile selecionado
                profile_metrics_available = [m for m in profile_metrics if m in df.columns]
                
                # Melhoria por categoria usando o final_metric_category_map
                for metric in profile_metrics_available:
                    if metric in final_metric_category_map and metric in recent_game.index and metric in historical_data.columns:
                        category = final_metric_category_map[metric]
                        recent_value = recent_game[metric]
                        avg_value = historical_data[metric].mean()
                        
                        if avg_value > 0:  # Evitar divisão por zero
                            improvement = ((recent_value - avg_value) / avg_value) * 100
                            
                            # Adicionar à categoria apropriada
                            if category == 'Ofensiva':
                                ofensiva_improvement = max(ofensiva_improvement, improvement)
                            elif category == 'Defensiva':
                                defensiva_improvement = max(defensiva_improvement, improvement)
                            elif category == 'Passe':
                                passe_improvement = max(passe_improvement, improvement)
                            elif category == 'Geral':
                                geral_improvement = max(geral_improvement, improvement)
                
                # Se não há métricas específicas do perfil, usar métricas padrão
                if ofensiva_improvement == 0 and 'Golos' in recent_game.index and 'Golos' in historical_data.columns:
                    recent_goals = recent_game['Golos']
                    avg_goals = historical_data['Golos'].mean()
                    ofensiva_improvement = ((recent_goals - avg_goals) / (avg_goals + 0.1)) * 100
                
                if defensiva_improvement == 0 and 'Intercepções' in recent_game.index and 'Intercepções' in historical_data.columns:
                    recent_def = recent_game['Intercepções']
                    avg_def = historical_data['Intercepções'].mean()
                    defensiva_improvement = ((recent_def - avg_def) / (avg_def + 0.1)) * 100
                
                if passe_improvement == 0 and 'Passes Totais Certos' in recent_game.index and 'Passes Totais Certos' in historical_data.columns:
                    recent_pass = recent_game['Passes Totais Certos']
                    avg_pass = historical_data['Passes Totais Certos'].mean()
                    passe_improvement = ((recent_pass - avg_pass) / (avg_pass + 0.1)) * 100
                
                if geral_improvement == 0 and 'Ações Totais' in recent_game.index and 'Ações Totais' in historical_data.columns:
                    recent_actions = recent_game['Ações Totais']
                    avg_actions = historical_data['Ações Totais'].mean()
                    geral_improvement = ((recent_actions - avg_actions) / (avg_actions + 0.1)) * 100
                
                # Extrair clube - usar os últimos 5 jogos para encontrar o mais frequente
                # Precisamos ter acesso a todos os dados do jogador, não apenas recent_game
                player_all_data = df[df['Jogo'] == recent_game.get('Jogo')]
                if player_all_data.empty:
                    player_all_data = df[df['Jogador'] == player]

                # Obter o clube mais frequente nos últimos 5 jogos
                clube = get_most_common_club_from_last_games(player_all_data, num_games=5, col_data_jogo=col_data_jogo)

                # Calcular score total com ênfase nas métricas do Position Profile
                total_score = (
                    max(ofensiva_improvement, 0) * 0.3 +
                    max(defensiva_improvement, 0) * 0.25 +
                    max(passe_improvement, 0) * 0.25 +
                    max(geral_improvement, 0) * 0.2
                )

                all_players_ranking.append({
                    'Jogador': player,
                    'Clube': clube,
                    'Posição': recent_game.get('Posição', 'N/A'),
                    'Score_Total': total_score,
                    'Melhoria_Ofensiva': ofensiva_improvement,
                    'Melhoria_Defensiva': defensiva_improvement,
                    'Melhoria_Passe': passe_improvement,
                    'Melhoria_Geral': geral_improvement,
                    'Golos_Ultimo': recent_game.get('Golos', 0),
                    'Assist_Ultimo': recent_game.get('Assistências', 0),
                    'Minutos_Ultimo': recent_game.get('Minutos', 0),
                    'Amarelos_Ultimo': recent_game.get('Cartões Amarelos', 0),
                    'Vermelhos_Ultimo': recent_game.get('Cartões Vermelhos', 0)
                })
            except Exception as e:
                continue
        
        # Criar DataFrame do ranking
        if all_players_ranking:
            ranking_df = pd.DataFrame(all_players_ranking)
            ranking_df = ranking_df.sort_values('Score_Total', ascending=False)
            
            # Exibir ranking
            st.markdown("### 🏅 Top 10 Jogadores em Melhoria")
            
            # Top 10 players
            top_10 = ranking_df.head(10)
            
            for idx, row in top_10.iterrows():
                rank = top_10.index.get_loc(idx) + 1
                
                # Determinar cor baseada na posição
                if rank <= 3:
                    color = "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)"
                    border = "3px solid #FFD700"
                elif rank <= 6:
                    color = "linear-gradient(135deg, #C0C0C0 0%, #808080 100%)"
                    border = "2px solid #C0C0C0"
                else:
                    color = "linear-gradient(135deg, #CD7F32 0%, #8B4513 100%)"
                    border = "2px solid #CD7F32"
                
                st.markdown(f"""
                <div style="
                    background: {color};
                    border-radius: 12px;
                    padding: 15px;
                    margin: 8px 0;
                    color: white;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    border: {border};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center;">
                            <div style="font-size: 2em; font-weight: bold; margin-right: 15px;">#{rank}</div>
                            <div>
                                <h3 style="margin: 0; font-size: 1.3em;">⚽ {row['Jogador']}</h3>
                                <p style="margin: 2px 0; font-size: 1em;">🏟️ {row['Clube']} | 📍 {row['Posição']}</p>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.8em; font-weight: bold;">{row['Score_Total']:.1f}</div>
                            <div style="font-size: 0.9em;">Score Total</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.9em;">
                        <div>⚽ Golos: {row['Golos_Ultimo']:.0f}</div>
                        <div>🅰️ Assist: {row['Assist_Ultimo']:.0f}</div>
                        <div>⏱️ Min: {row['Minutos_Ultimo']:.0f}</div>
                        <div>📊 Score: {row['Score_Total']:.1f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tabela completa do ranking
            with st.expander("📋 Ver Ranking Completo - Análise por Position Profile"):
                display_ranking = ranking_df.copy()
                display_ranking['Rank'] = range(1, len(display_ranking) + 1)
                
                # Renomear colunas para exibição
                display_ranking_display = display_ranking[['Rank', 'Jogador', 'Clube', 'Posição', 'Score_Total', 
                                                         'Melhoria_Ofensiva', 'Melhoria_Defensiva', 'Melhoria_Passe', 
                                                         'Melhoria_Geral']].copy()
                
                display_ranking_display.columns = ['#', 'Jogador', 'Clube', 'Posição', f'Score Total - {selected_profile}', 
                                                 'Melhoria Ofensiva (%)', 'Melhoria Defensiva (%)', 
                                                 'Melhoria Passe (%)', 'Melhoria Geral (%)']
                
                # Formatar números
                for col in [f'Score Total - {selected_profile}', 'Melhoria Ofensiva (%)', 'Melhoria Defensiva (%)', 
                           'Melhoria Passe (%)', 'Melhoria Geral (%)']:
                    display_ranking_display[col] = display_ranking_display[col].round(1)
                
                st.dataframe(
                    display_ranking_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download do ranking
                csv_data = display_ranking_display.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Ranking - {selected_profile} (CSV)",
                    data=csv_data,
                    file_name=f"ranking_jogadores_{selected_profile.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ Não foi possível gerar o ranking. Verifique se há dados suficientes para análise de melhoria.")
        
        if not all_players_ranking:
            st.warning("⚠️ Não foi possível gerar o ranking. Verifique se há dados suficientes para análise de melhoria.")
            st.warning("⚠️ Não foi possível gerar o ranking. Verifique se há dados suficientes.")
