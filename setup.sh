mkdir -p ~/.streamlit/

echo "\
[server]\n\n
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml