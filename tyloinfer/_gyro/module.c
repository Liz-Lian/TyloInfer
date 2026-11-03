
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "fit_core.inc"

static int floats(PyObject *object, Py_buffer *buffer, Py_ssize_t expected) {
    if (PyObject_GetBuffer(object, buffer, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return 0;
    if (buffer->itemsize != 4 || !buffer->format || strcmp(buffer->format, "f") ||
            buffer->len % 4 || (expected >= 0 && buffer->len != expected * 4)) {
        PyBuffer_Release(buffer); buffer->obj = NULL;
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float32 values with matching width");
        return 0;
    }
    const float *v = (const float *) buffer->buf;
    for (Py_ssize_t i=0; i<buffer->len/4; ++i) if (!isfinite(v[i])) {
        PyBuffer_Release(buffer); buffer->obj = NULL;
        PyErr_SetString(PyExc_ValueError, "weights and importance must be finite"); return 0;
    }
    return 1;
}

static PyObject *fit_row(PyObject *reference, PyObject *base, int type, PyObject *importance) {
    Py_buffer r={0}, b={0}, im={0};
    PyObject *result=NULL;
    float *target=NULL, *weights=NULL;
    if (!tq_type_k(type)) { PyErr_SetString(PyExc_ValueError,"unknown Gyro format"); return NULL; }
    if (!floats(reference,&r,-1)) goto done;
    const Py_ssize_t n=r.len/4;
    if (n<1 || n>PY_SSIZE_T_MAX/4-127) { PyErr_SetString(PyExc_ValueError,"invalid row width"); goto done; }
    if (base!=Py_None && !floats(base,&b,n)) goto done;
    if (importance!=Py_None && !floats(importance,&im,n)) goto done;
    const Py_ssize_t padded=(n+127)/128*128;
    target=(float *) calloc((size_t)padded,sizeof(float));
    if (im.obj || padded!=n) weights=(float *) calloc((size_t)padded,sizeof(float));
    if (!target || ((im.obj || padded!=n) && !weights)) { PyErr_NoMemory(); goto done; }
    for (Py_ssize_t i=0;i<n;++i) {
        target[i]=((const float *)r.buf)[i]-(b.obj?((const float *)b.buf)[i]:0.0f);
        if (!isfinite(target[i])) { PyErr_SetString(PyExc_ValueError,"residual exceeds float32 range"); goto done; }
        if (weights) {
            weights[i]=im.obj?((const float *)im.buf)[i]:1.0f;
            if (weights[i]<0) { PyErr_SetString(PyExc_ValueError,"importance must be nonnegative"); goto done; }
        }
    }
    const size_t bytes=tq_row_bytes(type,padded);
    if (bytes>PY_SSIZE_T_MAX) { PyErr_SetString(PyExc_OverflowError,"encoded row is too large"); goto done; }
    result=PyBytes_FromStringAndSize(NULL,(Py_ssize_t)bytes);
    if (!result) goto done;
    size_t written;
    Py_BEGIN_ALLOW_THREADS
    written=fit_target_row(type,target,PyBytes_AS_STRING(result),padded,weights);
    Py_END_ALLOW_THREADS
    if (written!=bytes) { Py_CLEAR(result); PyErr_NoMemory(); }
done:
    free(target);free(weights);
    if (im.obj) PyBuffer_Release(&im);
    if (b.obj) PyBuffer_Release(&b);
    if (r.obj) PyBuffer_Release(&r);
    return result;
}

static PyObject *fit_residual(PyObject *self, PyObject *args) {
    PyObject *reference,*base,*importance=Py_None;
    int type;
    if (!PyArg_ParseTuple(args,"OOi|O",&reference,&base,&type,&importance)) return NULL;
    if(base==Py_None) { PyErr_SetString(PyExc_ValueError,"residual fitting requires an immutable base row"); return NULL; }
    return fit_row(reference,base,type,importance);
}
static PyMethodDef methods[]={
    {"fit_residual",fit_residual,METH_VARARGS,"Fit one output neuron's residual while preserving its base."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_gyro_fit",NULL,-1,methods};
PyMODINIT_FUNC PyInit__gyro_fit(void) { return PyModule_Create(&module); }
