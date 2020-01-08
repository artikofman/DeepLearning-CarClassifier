def trailing_slash(st: str, use_back_slash: bool):
    if use_back_slash:
        if st.endswith("/"):
            return st
        else:
            return st + "/"
    else:
        if st.endswith("\\"):
            return st
        else:
            return st + "\\"
