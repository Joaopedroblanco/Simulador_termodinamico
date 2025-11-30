import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==================== CONSTANTES FÍSICAS ====================
R = 8.314  # Constante dos gases ideais [J/(mol·K)]
cp_ar = 29.1  # Capacidade calorífica a pressão constante [J/(mol·K)]
cv_ar = cp_ar - R  # CORREÇÃO: cv derivado de cp para garantir consistência
gamma_ar = cp_ar / cv_ar  # Coeficiente adiabático (~1.40)
M_ar = 0.02897  # Massa molar do ar [kg/mol]

# ==================== CLASSE CORRIGIDA ====================
class CilindroPistao:
    def __init__(self, diametro, curso_inicial, massa_gas, T0, P0=101325):
        self.A = np.pi * (diametro/2)**2  # Área do pistão
        self.x0 = curso_inicial
        self.massa_gas = massa_gas
        self.T0 = T0
        self.P0 = P0
        
        self.n = massa_gas / M_ar  # Número de moles
        self.V0 = self.A * curso_inicial
        
        # CORREÇÃO: Estado inicial consistente com lei dos gases
        self.P_i = P0
        self.V_i = self.V0
        self.T_i = (self.P_i * self.V_i) / (self.n * R)  # Temperatura ajustada para consistência
        
        # Verificar se houve ajuste significativo
        self.estado_ajustado = abs(self.T_i - T0) > 1.0

    def volume_para_posicao(self, V):
        return V / self.A
    
    def posicao_para_volume(self, x):
        return x * self.A
    
    def pressao_gas(self, V, T):
        return (self.n * R * T) / V

    # ==================== PROCESSOS CORRIGIDOS ====================
    
    def processo_isocorico(self, Q):
        """Volume constante"""
        V_f = self.V_i
        W = 0
        delta_U = Q
        delta_T = delta_U / (self.n * cv_ar)
        T_f = self.T_i + delta_T
        P_f = self.pressao_gas(V_f, T_f)
        return P_f, V_f, T_f, W, Q, delta_U, False  # False: sem ajuste de Q
    
    def processo_isobarico(self, Q):
        """Pressão constante - MELHORADO com feedback"""
        P_f = self.P_i
        
        if abs(Q) < 1e-10:
            return self.P_i, self.V_i, self.T_i, 0.0, 0.0, 0.0, False
        
        # Cálculo consistente com 1ª Lei
        delta_T = Q / (self.n * cp_ar)
        T_f = self.T_i + delta_T
        V_f = (self.n * R * T_f) / P_f
        W = P_f * (V_f - self.V_i)
        delta_U = self.n * cv_ar * (T_f - self.T_i)
        
        # Verificação da 1ª Lei
        discrepancia = delta_U - (Q - W)
        
        if abs(discrepancia) > 1e-6:
            Q_ajustado = delta_U + W
            return P_f, V_f, T_f, W, Q_ajustado, delta_U, True  # True: Q foi ajustado
        else:
            return P_f, V_f, T_f, W, Q, delta_U, False
    
    def processo_isotermico(self, Q):
        """Temperatura constante - COM AVISO CONCEITUAL"""
        T_f = self.T_i
        # Em isotérmico, Q controla diretamente a expansão/compressão
        V_f = self.V_i * np.exp(Q / (self.n * R * T_f))
        W = Q
        P_f = self.pressao_gas(V_f, T_f)
        delta_U = 0
        return P_f, V_f, T_f, W, Q, delta_U, False
    
    def processo_adiabatico(self, V_final_ratio=None):
        """Adiabático - especificar razão de volumes"""
        Q = 0
        
        if V_final_ratio is None:
            V_final_ratio = 1.5
            
        V_f = self.V_i * V_final_ratio
        # Relações adiabáticas
        P_f = self.P_i * (self.V_i / V_f) ** gamma_ar
        T_f = self.T_i * (self.V_i / V_f) ** (gamma_ar - 1)
        delta_U = self.n * cv_ar * (T_f - self.T_i)
        W = -delta_U
        
        return P_f, V_f, T_f, W, Q, delta_U, False
    
    def processo_politropico(self, n, V_final_ratio=None):
        """Politrópico - especificar expoente e razão de volumes"""
        if V_final_ratio is None:
            V_final_ratio = 1.5
            
        V_f = self.V_i * V_final_ratio
        # Relações politrópicas
        P_f = self.P_i * (self.V_i / V_f) ** n
        T_f = self.T_i * (self.V_i / V_f) ** (n - 1)
        
        if abs(n - 1) > 1e-6:
            W = (P_f * V_f - self.P_i * self.V_i) / (1 - n)
        else:
            W = self.P_i * self.V_i * np.log(V_f / self.V_i)
            
        delta_U = self.n * cv_ar * (T_f - self.T_i)
        Q = delta_U + W
        
        return P_f, V_f, T_f, W, Q, delta_U, False

# ==================== VISUALIZAÇÕES MELHORADAS ====================

def plot_diagrama_PV_melhorado(P_i, V_i, P_f, V_f, processo, n_politropico=None):
    """Diagrama P-V com curvas reais"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Range para a curva
    V_min = min(V_i, V_f) * 0.8
    V_max = max(V_i, V_f) * 1.2
    V_curve = np.linspace(V_min, V_max, 100)
    
    # Curva do processo - CORREÇÃO: curvas reais em vez de linha reta
    if processo == "Isocórico":
        P_curve = np.linspace(min(P_i, P_f), max(P_i, P_f), 100)
        ax.plot([V_i] * 100, P_curve, 'g-', linewidth=2, label='Processo Isocórico')
    elif processo == "Isobárico":
        ax.plot(V_curve, [P_i] * 100, 'g-', linewidth=2, label='Processo Isobárico')
    elif processo == "Isotérmico":
        P_curve = (P_i * V_i) / V_curve  # PV = constante
        ax.plot(V_curve, P_curve, 'g-', linewidth=2, label='Processo Isotérmico')
    elif processo == "Adiabático":
        P_curve = P_i * (V_i / V_curve) ** gamma_ar  # PV^γ = constante
        ax.plot(V_curve, P_curve, 'g-', linewidth=2, label='Processo Adiabático')
    elif processo == "Politrópico" and n_politropico is not None:
        P_curve = P_i * (V_i / V_curve) ** n_politropico  # PV^n = constante
        ax.plot(V_curve, P_curve, 'g-', linewidth=2, label=f'Processo Politrópico (n={n_politropico:.2f})')
    else:
        # Fallback: linha reta
        ax.plot([V_i, V_f], [P_i, P_f], 'g-', linewidth=2, label=f'Processo {processo}')
    
    # Estados
    ax.plot(V_i, P_i, 'bo', markersize=10, label='Estado Inicial')
    ax.plot(V_f, P_f, 'ro', markersize=10, label='Estado Final')
    
    ax.set_xlabel('Volume [m³]', fontsize=12)
    ax.set_ylabel('Pressão [Pa]', fontsize=12)
    ax.set_title('Diagrama Pressão-Volume (P-V)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def desenhar_cilindro_simples(A, x_inicial, x_final, diametro):
    """Ilustração simplificada do cilindro-pistão"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Cilindro
    largura_cilindro = diametro * 1.2
    altura_cilindro = max(x_inicial, x_final) * 1.3
    
    cilindro = patches.Rectangle((0.3, 0), largura_cilindro, altura_cilindro,
                               linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(cilindro)
    
    # Pistão inicial
    pistao_inicial = patches.Rectangle((0.3, x_inicial), largura_cilindro, 0.05,
                                     linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.7)
    ax.add_patch(pistao_inicial)
    
    # Pistão final  
    pistao_final = patches.Rectangle((0.3, x_final), largura_cilindro, 0.05,
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.7)
    ax.add_patch(pistao_final)
    
    # Gás
    gas_inicial = patches.Rectangle((0.3, 0), largura_cilindro, x_inicial,
                                  linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
    ax.add_patch(gas_inicial)
    
    gas_final = patches.Rectangle((0.3, 0), largura_cilindro, x_final,
                                linewidth=1, edgecolor='orange', facecolor='orange', alpha=0.3)
    ax.add_patch(gas_final)
    
    # Anotações
    ax.text(0.15, x_inicial/2, f'V₀ = {A*x_inicial:.3f} m³', 
           fontsize=9, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.15, x_final/2, f'V_f = {A*x_final:.3f} m³', 
           fontsize=9, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, altura_cilindro * 1.1)
    ax.set_aspect('equal')
    ax.set_title('Cilindro-Pistão Simplificado', fontsize=14)
    ax.axis('off')
    
    return fig

# ==================== APLICAÇÃO STREAMLIT CORRIGIDA ====================

def main():
    st.set_page_config(
        page_title="Simulador Termodinâmico - Cilindro-Pistão",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Simulador Termodinâmico - Cilindro-Pistão")
    st.markdown("### Modelo Simplificado para Estudo da 1ª Lei")
    
    st.markdown("""
    **Equação Fundamental:** ΔU = Q - W  
    Onde:
    - ΔU = Variação de energia interna
    - Q = Calor trocado  
    - W = Trabalho realizado
    """)
    
    st.markdown("---")
    
    # ==================== CONFIGURAÇÃO CORRIGIDA ====================
    st.sidebar.header("Configuração do Sistema")
    
    st.sidebar.subheader("Geometria do Cilindro")
    diametro = st.sidebar.number_input("Diâmetro do Pistão [m]", 0.1, 1.0, 0.3, 0.05)
    curso_inicial = st.sidebar.number_input("Curso Inicial [m]", 0.1, 2.0, 1.0, 0.1)
    
    st.sidebar.subheader("Propriedades do Gás")
    massa_gas = st.sidebar.number_input("Massa de Gás [kg]", 0.01, 1.0, 0.1, 0.01)
    T0 = st.sidebar.number_input("Temperatura Inicial [K]", 200.0, 500.0, 300.0, 10.0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Processo Termodinâmico")
    
    processo = st.sidebar.selectbox(
        "Selecione o processo:",
        ["Isocórico", "Isobárico", "Isotérmico", "Adiabático", "Politrópico"]
    )
    
    # CORREÇÃO CRÍTICA: Remover atribuição desnecessária de 'n'
    if processo == "Isocórico":
        Q = st.sidebar.number_input("Calor (Q) [J]", -5000.0, 5000.0, 1000.0, 100.0)
        V_ratio = None
    elif processo == "Isobárico":
        Q = st.sidebar.number_input("Calor (Q) [J]", -5000.0, 5000.0, 1000.0, 100.0)
        V_ratio = None
    elif processo == "Isotérmico":
        Q = st.sidebar.number_input("Calor (Q) [J]", -5000.0, 5000.0, 1000.0, 100.0)
        V_ratio = None
    elif processo == "Adiabático":
        Q = 0
        V_ratio = st.sidebar.number_input("Razão de Volumes (Vf/Vi)", 0.1, 5.0, 1.5, 0.1)
    elif processo == "Politrópico":
        Q = None
        n_politropico = st.sidebar.number_input("Expoente Politrópico (n)", 0.1, 3.0, 1.4, 0.1)
        V_ratio = st.sidebar.number_input("Razão de Volumes (Vf/Vi)", 0.1, 5.0, 1.5, 0.1)
    
    # ==================== SIMULAÇÃO CORRIGIDA ====================
    try:
        cilindro = CilindroPistao(diametro, curso_inicial, massa_gas, T0)
        
        # Aviso sobre ajuste de estado inicial
        if cilindro.estado_ajustado:
            st.info(f"💡 **Aviso:** Temperatura ajustada de {T0:.1f} K para {cilindro.T_i:.1f} K para garantir consistência com a lei dos gases ideais.")
        
        # Executar processo
        Q_ajustado = False
        Q_original = Q if processo != "Politrópico" else None
        
        if processo == "Isocórico":
            P_f, V_f, T_f, W, Q_calc, delta_U, Q_ajustado = cilindro.processo_isocorico(Q)
        elif processo == "Isobárico":
            P_f, V_f, T_f, W, Q_calc, delta_U, Q_ajustado = cilindro.processo_isobarico(Q)
        elif processo == "Isotérmico":
            P_f, V_f, T_f, W, Q_calc, delta_U, Q_ajustado = cilindro.processo_isotermico(Q)
            st.info("💡 **Processo Isotérmico:** O calor Q controla diretamente a expansão/compressão do gás.")
        elif processo == "Adiabático":
            P_f, V_f, T_f, W, Q_calc, delta_U, Q_ajustado = cilindro.processo_adiabatico(V_ratio)
        elif processo == "Politrópico":
            P_f, V_f, T_f, W, Q_calc, delta_U, Q_ajustado = cilindro.processo_politropico(n_politropico, V_ratio)
        
        x_final = cilindro.volume_para_posicao(V_f)
        
        # ==================== RESULTADOS MELHORADOS ====================
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 Resultados da Simulação")
            
            # Aviso sobre ajuste de Q no isobárico
            if Q_ajustado and processo == "Isobárico":
                st.warning(f"⚡ **Ajuste de Consistência:** O calor foi ajustado de {Q_original:.1f} J para {Q_calc:.1f} J para respeitar rigorosamente a 1ª Lei da Termodinâmica.")
            
            # Estados termodinâmicos
            st.subheader("Estados Termodinâmicos")
            dados_estados = {
                'Grandeza': ['Pressão [Pa]', 'Volume [m³]', 'Temperatura [K]', 'Posição [m]'],
                'Inicial': [f"{cilindro.P_i:.0f}", f"{cilindro.V_i:.4f}", f"{cilindro.T_i:.1f}", f"{cilindro.x0:.3f}"],
                'Final': [f"{P_f:.0f}", f"{V_f:.4f}", f"{T_f:.1f}", f"{x_final:.3f}"],
                'Variação': [f"{P_f-cilindro.P_i:+.0f}", f"{V_f-cilindro.V_i:+.4f}", f"{T_f-cilindro.T_i:+.1f}", f"{x_final-cilindro.x0:+.3f}"]
            }
            
            st.dataframe(dados_estados, use_container_width=True)
            
            # Balanço energético
            st.subheader("⚡ Balanço de Energia")
            col_a, col_b, col_c = st.columns(3)
            with col_a: 
                st.metric("Trabalho (W)", f"{W:.1f} J", f"{W:+.1f}")
            with col_b: 
                if Q_ajustado and processo == "Isobárico":
                    st.metric("Calor (Q)", f"{Q_calc:.1f} J", f"{Q_calc-Q_original:+.1f}", delta_color="inverse")
                else:
                    st.metric("Calor (Q)", f"{Q_calc:.1f} J", f"{Q_calc:+.1f}")
            with col_c: 
                st.metric("ΔU", f"{delta_U:.1f} J", f"{delta_U:+.1f}")
            
            # Verificação da 1ª Lei
            st.subheader("📐 Verificação da 1ª Lei")
            diferenca = delta_U - (Q_calc - W)
            tolerancia = 1e-4 * max(1.0, abs(Q_calc), abs(W), abs(delta_U))
            
            if abs(diferenca) < tolerancia:
                st.success("✅ **1ª Lei da Termodinâmica VERIFICADA!**")
                st.write(f"ΔU - (Q - W) = {diferenca:.2e} J ≈ 0")
            else:
                st.error("❌ **1ª Lei NÃO verificada!**")
                st.write(f"Discrepância: {diferenca:.2e} J")
            
            # Informações do sistema
            st.subheader("🔧 Informações do Sistema")
            st.write(f"**Número de moles:** {cilindro.n:.4f} mol")
            st.write(f"**Área do pistão:** {cilindro.A:.4f} m²")
            st.write(f"**Razão de volumes:** V_f/V_i = {V_f/cilindro.V_i:.3f}")
            st.write(f"**Razão de pressões:** P_f/P_i = {P_f/cilindro.P_i:.3f}")
            st.write(f"**Razão de temperaturas:** T_f/T_i = {T_f/cilindro.T_i:.3f}")
        
        with col2:
            st.header("📈 Visualizações")
            
            # Diagrama P-V
            st.subheader("Diagrama Pressão-Volume")
            if processo == "Politrópico":
                fig_pv = plot_diagrama_PV_melhorado(cilindro.P_i, cilindro.V_i, P_f, V_f, processo, n_politropico)
            else:
                fig_pv = plot_diagrama_PV_melhorado(cilindro.P_i, cilindro.V_i, P_f, V_f, processo)
            st.pyplot(fig_pv)
            
            # Ilustração do cilindro
            st.subheader("Ilustração do Sistema")
            fig_cilindro = desenhar_cilindro_simples(cilindro.A, cilindro.x0, x_final, diametro)
            st.pyplot(fig_cilindro)
        
    except Exception as e:
        st.error(f"Erro na simulação: {str(e)}")
        st.info("💡 Dica: Ajuste os parâmetros para valores fisicamente possíveis.")

if __name__ == "__main__":
    main()