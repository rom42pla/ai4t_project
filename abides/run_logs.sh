#



# Plot mid prices with means
cd util/plotting && python3 -u liquidity_telemetry_mean.py -s ../../log/twosym2/ExchangeAgent0.bz2 -s ../../log/twosym3/ExchangeAgent0.bz2 -s ../../log/twosym6/ExchangeAgent0.bz2 -s ../../log/twosym5/ExchangeAgent0.bz2 -s ../../log/twosym7/ExchangeAgent0.bz2 -s ../../log/twosym4/ExchangeAgent0.bz2 \
    -b ../../log/twosym2/ORDERBOOK_SYM1_FULL.bz2 -b ../../log/twosym3/ORDERBOOK_SYM1_FULL.bz2 -b ../../log/twosym6/ORDERBOOK_SYM1_FULL.bz2 -b ../../log/twosym5/ORDERBOOK_SYM1_FULL.bz2  -b ../../log/twosym7/ORDERBOOK_SYM1_FULL.bz2 -b ../../log/twosym4/ORDERBOOK_SYM1_FULL.bz2 \
    -o twosym_all_sy1.png -c configs/plot_2sym.json && cd ../../ &


cd util/plotting && python3 -u liquidity_telemetry_mean.py -s ../../log/twosym2/ExchangeAgent0.bz2 -s ../../log/twosym3/ExchangeAgent0.bz2 -s ../../log/twosym6/ExchangeAgent0.bz2 -s ../../log/twosym5/ExchangeAgent0.bz2 -s ../../log/twosym7/ExchangeAgent0.bz2 -s ../../log/twosym4/ExchangeAgent0.bz2 \
    -b ../../log/twosym2/ORDERBOOK_SYM2_FULL.bz2 -b ../../log/twosym3/ORDERBOOK_SYM2_FULL.bz2 -b ../../log/twosym6/ORDERBOOK_SYM2_FULL.bz2 -b ../../log/twosym5/ORDERBOOK_SYM2_FULL.bz2  -b ../../log/twosym7/ORDERBOOK_SYM2_FULL.bz2 -b ../../log/twosym4/ORDERBOOK_SYM2_FULL.bz2 \
    -o twosym_all_sy2.png -c configs/plot_2sym.json && cd ../../ &

cd util/plotting && python3 -u liquidity_telemetry_mean.py -s ../../log/twosym2/ExchangeAgent0.bz2 -s ../../log/twosym3/ExchangeAgent0.bz2 -s ../../log/twosym6/ExchangeAgent0.bz2 -s ../../log/twosym5/ExchangeAgent0.bz2 -s ../../log/twosym7/ExchangeAgent0.bz2 -s ../../log/twosym4/ExchangeAgent0.bz2 \
    -b ../../log/twosym2/ORDERBOOK_ETF_FULL.bz2 -b ../../log/twosym3/ORDERBOOK_ETF_FULL.bz2 -b ../../log/twosym6/ORDERBOOK_ETF_FULL.bz2 -b ../../log/twosym5/ORDERBOOK_ETF_FULL.bz2  -b ../../log/twosym7/ORDERBOOK_ETF_FULL.bz2 -b ../../log/twosym4/ORDERBOOK_ETF_FULL.bz2 \
    -o twosym_all_etf.png -c configs/plot_2sym.json && cd ../../

# Plot single element
for f in "2" "3" "4" "5" "6" "7" "8";
do
    cd util/plotting && python3 -u liquidity_telemetry.py ../../log/twosym${f}/ExchangeAgent0.bz2 ../../log/twosym${f}/ORDERBOOK_SYM1_FULL.bz2 \
    -o twosym${f}_sy1.png -c configs/plot_2sym.json && cd ../../ &

    cd util/plotting && python3 -u liquidity_telemetry.py ../../log/twosym${f}/ExchangeAgent0.bz2 ../../log/twosym${f}/ORDERBOOK_SYM2_FULL.bz2 \
    -o twosym${f}_sy2.png -c configs/plot_2sym.json && cd ../../ &

    cd util/plotting && python3 -u liquidity_telemetry.py ../../log/twosym${f}/ExchangeAgent0.bz2 ../../log/twosym${f}/ORDERBOOK_ETF_FULL.bz2 \
    -o twosym${f}_etf.png -c configs/plot_2sym.json && cd ../../ &
done;
