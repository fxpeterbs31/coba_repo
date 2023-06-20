import streamlit as st

# Fungsi untuk validasi email
def validate_email(email):
    if "@" not in email or ".com" not in email:
        return False
    return True

# Fungsi untuk validasi password
def validate_password(password):
    has_letter = False
    has_digit = False
    for char in password:
        if char.isalpha():
            has_letter = True
        elif char.isdigit():
            has_digit = True
    if has_letter and has_digit:
        return True
    return False

# Daftar pengguna yang sudah terdaftar
registered_users = []

# Layar Daftar
def register_screen():
    st.title("Daftar")
    name = st.text_input("Nama")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Daftar"):
        if name == "":
            st.error("Nama tidak boleh kosong!")
        elif not validate_email(email):
            st.error("Email harus memiliki tanda @ dan .com!")
        elif not validate_password(password):
            st.error("Password harus mengandung huruf dan angka!")
        else:
            registered_users.append({
                "name": name,
                "email": email,
                "password": password
            })
            st.success("Pendaftaran berhasil! Silakan login.")
            login_screen()

    st.markdown("Sudah punya akun? Silakan [login](login).")

# Layar Login
def login_screen():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        found = False
        for user in registered_users:
            if user["email"] == email and user["password"] == password:
                found = True
                break
        if found:
            st.success("Login berhasil!")
        else:
            st.error("Email atau password salah!")

    st.markdown("Belum punya akun? Silakan [daftar](register).")

# Menampilkan layar Daftar saat aplikasi dijalankan
register_screen()
