use {
    rustls::pki_types::{CertificateDer, PrivateKeyDer},
    solana_keypair::Keypair,
    solana_tls_utils::new_dummy_x509_certificate,
};

pub struct QuicClientCertificate {
    pub certificate: CertificateDer<'static>,
    pub key: PrivateKeyDer<'static>,
}

impl Default for QuicClientCertificate {
    fn default() -> Self {
        QuicClientCertificate::new(&Keypair::new())
    }
}

impl QuicClientCertificate {
    pub fn new(keypair: &Keypair) -> Self {
        let (certificate, key) = new_dummy_x509_certificate(keypair);
        Self { certificate, key }
    }
}
