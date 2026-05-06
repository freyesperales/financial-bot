"""
Indicadores Avanzados para Trading Bot
Fibonacci, Ichimoku, ADX, Volume Profile, Patrones de Velas
"""
import pandas as pd
import numpy as np
import pandas_ta as ta


class AdvancedIndicators:
    """Clase con indicadores técnicos avanzados"""

    @staticmethod
    def fibonacci_retracements(df, period=20):
        """
        Calcula niveles de Fibonacci basados en últimos máximos/mínimos

        Niveles: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%

        Returns:
            dict con niveles y señales
        """
        recent = df.tail(period)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low

        levels = {
            'fib_0': high,
            'fib_23.6': high - (diff * 0.236),
            'fib_38.2': high - (diff * 0.382),
            'fib_50': high - (diff * 0.5),
            'fib_61.8': high - (diff * 0.618),
            'fib_78.6': high - (diff * 0.786),
            'fib_100': low
        }

        current_price = df['close'].iloc[-1]

        # Determinar señal
        signal = None
        if current_price <= levels['fib_61.8'] and current_price >= levels['fib_78.6']:
            signal = "COMPRA"  # Zona de rebote
        elif current_price >= levels['fib_23.6']:
            signal = "VENTA"   # Zona de resistencia

        return {
            'levels': levels,
            'signal': signal,
            'current': current_price
        }

    @staticmethod
    def ichimoku_cloud(df):
        """
        Ichimoku Cloud (Nube de Ichimoku)

        Componentes:
        - Tenkan-sen (línea de conversión): (9-period high + low) / 2
        - Kijun-sen (línea base): (26-period high + low) / 2
        - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, desplazado 26 períodos
        - Senkou Span B (Leading Span B): (52-period high + low) / 2, desplazado 26 períodos
        - Chikou Span (Lagging Span): Precio de cierre, desplazado -26 períodos

        Returns:
            DataFrame con componentes de Ichimoku y señal
        """
        # Calcular componentes
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)

        chikou_span = df['close'].shift(-26)

        # Añadir al DataFrame
        result = df.copy()
        result['tenkan_sen'] = tenkan_sen
        result['kijun_sen'] = kijun_sen
        result['senkou_span_a'] = senkou_span_a
        result['senkou_span_b'] = senkou_span_b
        result['chikou_span'] = chikou_span

        # Determinar señal
        current_price = df['close'].iloc[-1]
        current_tenkan = tenkan_sen.iloc[-1]
        current_kijun = kijun_sen.iloc[-1]
        current_span_a = senkou_span_a.iloc[-1] if not pd.isna(senkou_span_a.iloc[-1]) else 0
        current_span_b = senkou_span_b.iloc[-1] if not pd.isna(senkou_span_b.iloc[-1]) else 0

        signal = None
        if current_tenkan > current_kijun and current_price > max(current_span_a, current_span_b):
            signal = "COMPRA"  # Tendencia alcista, precio sobre la nube
        elif current_tenkan < current_kijun and current_price < min(current_span_a, current_span_b):
            signal = "VENTA"   # Tendencia bajista, precio bajo la nube

        return {
            'data': result,
            'signal': signal,
            'tenkan': current_tenkan,
            'kijun': current_kijun
        }

    @staticmethod
    def adx(df, length=14):
        """
        ADX (Average Directional Index)
        Mide la fuerza de la tendencia

        Valores:
        - ADX < 20: Tendencia débil
        - ADX 20-25: Tendencia moderada
        - ADX 25-50: Tendencia fuerte
        - ADX > 50: Tendencia muy fuerte

        Returns:
            dict con ADX, +DI, -DI y señal
        """
        # Calcular usando pandas_ta
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=length)

        if adx_data is None or adx_data.empty:
            return None

        adx_value = adx_data[f'ADX_{length}'].iloc[-1]
        plus_di = adx_data[f'DMP_{length}'].iloc[-1]
        minus_di = adx_data[f'DMN_{length}'].iloc[-1]

        # Determinar señal
        signal = None
        trend_strength = "débil"

        if adx_value > 25:
            trend_strength = "fuerte"
            if plus_di > minus_di:
                signal = "COMPRA"  # Tendencia alcista fuerte
            else:
                signal = "VENTA"   # Tendencia bajista fuerte
        elif adx_value > 20:
            trend_strength = "moderada"

        return {
            'adx': adx_value,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'signal': signal,
            'trend_strength': trend_strength
        }

    @staticmethod
    def stochastic_rsi(df, length=14, rsi_length=14, k=3, d=3):
        """
        Stochastic RSI
        Oscilador que aplica la fórmula Stochastic al RSI

        Valores:
        - > 80: Sobrecomprado
        - < 20: Sobrevendido

        Returns:
            dict con StochRSI K, D y señal
        """
        # Calcular RSI
        rsi = ta.rsi(df['close'], length=rsi_length)

        if rsi is None:
            return None

        # Calcular Stochastic del RSI
        rsi_min = rsi.rolling(window=length).min()
        rsi_max = rsi.rolling(window=length).max()

        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        # Calcular %K y %D
        k_line = stoch_rsi.rolling(window=k).mean()
        d_line = k_line.rolling(window=d).mean()

        k_value = k_line.iloc[-1]
        d_value = d_line.iloc[-1]

        # Determinar señal
        signal = None
        if k_value < 20 and k_value > d_value:
            signal = "COMPRA"  # Cruce alcista en zona de sobreventa
        elif k_value > 80 and k_value < d_value:
            signal = "VENTA"   # Cruce bajista en zona de sobrecompra

        return {
            'k': k_value,
            'd': d_value,
            'signal': signal
        }

    @staticmethod
    def volume_profile(df, bins=20):
        """
        Volume Profile
        Identifica niveles de precio con mayor volumen

        Returns:
            dict con niveles de soporte/resistencia basados en volumen
        """
        # Agrupar precio en bins
        df_copy = df.copy()
        df_copy['price_bin'] = pd.cut(df_copy['close'], bins=bins)

        # Sumar volumen por bin
        volume_by_price = df_copy.groupby('price_bin')['volume'].sum()

        # Encontrar POC (Point of Control) - precio con mayor volumen
        poc_bin = volume_by_price.idxmax()
        poc_price = poc_bin.mid

        # Encontrar VAH (Value Area High) y VAL (Value Area Low)
        # Aproximadamente 70% del volumen
        total_volume = volume_by_price.sum()
        target_volume = total_volume * 0.7

        sorted_volume = volume_by_price.sort_values(ascending=False)
        cumsum = 0
        value_area_bins = []

        for bin_range, vol in sorted_volume.items():
            cumsum += vol
            value_area_bins.append(bin_range)
            if cumsum >= target_volume:
                break

        # Calcular VAH y VAL
        vah = max([b.right for b in value_area_bins])
        val = min([b.left for b in value_area_bins])

        current_price = df['close'].iloc[-1]

        # Determinar señal
        signal = None
        if current_price < val:
            signal = "COMPRA"  # Precio bajo value area
        elif current_price > vah:
            signal = "VENTA"   # Precio sobre value area

        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'signal': signal
        }

    @staticmethod
    def detect_candlestick_patterns(df):
        """
        Detecta patrones de velas japonesas

        Patrones:
        - Hammer / Hanging Man
        - Engulfing (alcista/bajista)
        - Doji
        - Morning Star / Evening Star

        Returns:
            list de patrones detectados
        """
        patterns = []

        if len(df) < 3:
            return patterns

        # Última vela
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        has_prev2 = len(df) >= 3

        # Calcular body y sombras
        body = abs(current['close'] - current['open'])
        full_range = current['high'] - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])
        lower_shadow = min(current['open'], current['close']) - current['low']

        # 1. HAMMER (alcista)
        if (lower_shadow > body * 2 and
            upper_shadow < body * 0.3 and
            current['close'] < prev['close']):
            patterns.append({"pattern": "Hammer", "signal": "COMPRA", "strength": "fuerte"})

        # 2. HANGING MAN (bajista)
        if (lower_shadow > body * 2 and
            upper_shadow < body * 0.3 and
            current['close'] > prev['close']):
            patterns.append({"pattern": "Hanging Man", "signal": "VENTA", "strength": "fuerte"})

        # 3. BULLISH ENGULFING (alcista)
        if (prev['close'] < prev['open'] and
            current['close'] > current['open'] and
            current['open'] < prev['close'] and
            current['close'] > prev['open']):
            patterns.append({"pattern": "Bullish Engulfing", "signal": "COMPRA", "strength": "muy fuerte"})

        # 4. BEARISH ENGULFING (bajista)
        if (prev['close'] > prev['open'] and
            current['close'] < current['open'] and
            current['open'] > prev['close'] and
            current['close'] < prev['open']):
            patterns.append({"pattern": "Bearish Engulfing", "signal": "VENTA", "strength": "muy fuerte"})

        # 5. DOJI (indecisión)
        if body < (full_range * 0.1):
            patterns.append({"pattern": "Doji", "signal": "NEUTRO", "strength": "débil"})

        # 6. MORNING STAR (alcista)
        if (has_prev2 and
            prev2['close'] < prev2['open'] and  # Primera vela roja
            abs(prev['close'] - prev['open']) < body * 0.3 and  # Segunda vela pequeña (doji)
            current['close'] > current['open'] and  # Tercera vela verde
            current['close'] > prev2['open']):  # Cierra sobre apertura de primera
            patterns.append({"pattern": "Morning Star", "signal": "COMPRA", "strength": "muy fuerte"})

        # 7. EVENING STAR (bajista)
        if (has_prev2 and
            prev2['close'] > prev2['open'] and  # Primera vela verde
            abs(prev['close'] - prev['open']) < body * 0.3 and  # Segunda vela pequeña
            current['close'] < current['open'] and  # Tercera vela roja
            current['close'] < prev2['open']):  # Cierra bajo apertura de primera
            patterns.append({"pattern": "Evening Star", "signal": "VENTA", "strength": "muy fuerte"})

        return patterns


# ─────────────────────────────────────────────
#  EJEMPLO DE USO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("  📊 Indicadores Avanzados - Test")
    print("=" * 60)
    print()

    # Descargar datos de prueba
    print("Descargando datos de AMD...")
    df = yf.download("AMD", period="1mo", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.lower() for c in df.columns]

    print(f"✅ {len(df)} días descargados")
    print()

    # Crear instancia
    indicators = AdvancedIndicators()

    # 1. Fibonacci
    print("1. Fibonacci Retracements:")
    fib = indicators.fibonacci_retracements(df)
    print(f"   Precio actual: ${fib['current']:.2f}")
    for level, price in fib['levels'].items():
        print(f"   {level}: ${price:.2f}")
    print(f"   Señal: {fib['signal']}")
    print()

    # 2. Ichimoku
    print("2. Ichimoku Cloud:")
    ichimoku = indicators.ichimoku_cloud(df)
    print(f"   Tenkan-sen: ${ichimoku['tenkan']:.2f}")
    print(f"   Kijun-sen: ${ichimoku['kijun']:.2f}")
    print(f"   Señal: {ichimoku['signal']}")
    print()

    # 3. ADX
    print("3. ADX (Fuerza de Tendencia):")
    adx = indicators.adx(df)
    if adx:
        print(f"   ADX: {adx['adx']:.2f}")
        print(f"   +DI: {adx['plus_di']:.2f}")
        print(f"   -DI: {adx['minus_di']:.2f}")
        print(f"   Fuerza: {adx['trend_strength']}")
        print(f"   Señal: {adx['signal']}")
    print()

    # 4. Stochastic RSI
    print("4. Stochastic RSI:")
    stoch = indicators.stochastic_rsi(df)
    if stoch:
        print(f"   %K: {stoch['k']:.2f}")
        print(f"   %D: {stoch['d']:.2f}")
        print(f"   Señal: {stoch['signal']}")
    print()

    # 5. Volume Profile
    print("5. Volume Profile:")
    vol_profile = indicators.volume_profile(df)
    print(f"   POC (Point of Control): ${vol_profile['poc']:.2f}")
    print(f"   VAH (Value Area High): ${vol_profile['vah']:.2f}")
    print(f"   VAL (Value Area Low): ${vol_profile['val']:.2f}")
    print(f"   Señal: {vol_profile['signal']}")
    print()

    # 6. Patrones de Velas
    print("6. Patrones de Velas:")
    patterns = indicators.detect_candlestick_patterns(df)
    if patterns:
        for p in patterns:
            print(f"   ✨ {p['pattern']} - {p['signal']} ({p['strength']})")
    else:
        print("   Sin patrones detectados")

    print()
    print("=" * 60)
