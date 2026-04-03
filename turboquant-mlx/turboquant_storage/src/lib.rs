use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use numpy::ndarray::Array1;
use pyo3::prelude::*;

/// Packs an array of 8-bit integers (values 0-7) into an array of 8-bit integers 
/// where each byte contains 3 bits. So 8 elements take 24 bits = 3 bytes.
#[pyfunction]
fn pack_3bit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> &'py PyArray1<u8> {
    let x_arr = x.as_array();
    let n = x_arr.len();
    
    let num_chunks = (n + 7) / 8;
    let out_len = num_chunks * 3;
    let mut out = Array1::<u8>::zeros(out_len);
    
    for i in 0..num_chunks {
        let mut vals = [0u8; 8];
        for j in 0..8 {
            let idx = i * 8 + j;
            if idx < n {
                vals[j] = x_arr[idx] & 0x07;
            }
        }
        
        out[i * 3]     = (vals[0] << 5) | (vals[1] << 2) | (vals[2] >> 1);
        out[i * 3 + 1] = ((vals[2] & 0x01) << 7) | (vals[3] << 4) | (vals[4] << 1) | (vals[5] >> 2);
        out[i * 3 + 2] = ((vals[5] & 0x03) << 6) | (vals[6] << 3) | vals[7];
    }
    
    out.into_pyarray(py)
}

/// Unpacks an array of 8-bit integers containing 3-bit values into an array of 8-bit integers.
#[pyfunction]
fn unpack_3bit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
    original_len: usize,
) -> &'py PyArray1<u8> {
    let x_arr = x.as_array();
    let num_chunks = x_arr.len() / 3;
    let mut out = Array1::<u8>::zeros(num_chunks * 8);
    
    for i in 0..num_chunks {
        let b0 = x_arr[i * 3];
        let b1 = x_arr[i * 3 + 1];
        let b2 = x_arr[i * 3 + 2];
        
        out[i * 8]     = (b0 >> 5) & 0x07;
        out[i * 8 + 1] = (b0 >> 2) & 0x07;
        out[i * 8 + 2] = ((b0 << 1) & 0x06) | ((b1 >> 7) & 0x01);
        out[i * 8 + 3] = (b1 >> 4) & 0x07;
        out[i * 8 + 4] = (b1 >> 1) & 0x07;
        out[i * 8 + 5] = ((b1 << 2) & 0x04) | ((b2 >> 6) & 0x03);
        out[i * 8 + 6] = (b2 >> 3) & 0x07;
        out[i * 8 + 7] = b2 & 0x07;
    }
    
    let out_slice = out.slice(numpy::ndarray::s![..original_len]).to_owned();
    out_slice.into_pyarray(py)
}

#[pymodule]
fn turboquant_storage(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_3bit, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_3bit, m)?)?;
    Ok(())
}