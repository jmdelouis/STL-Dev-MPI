Python: ``stl_port.native``
===========================

The Python bindings of the native engines (``python/stl_ensemble/stl_port/native.py``), through the
C ABI of ``crates/stl-bridge``. Build the library with
``cargo build --release -p stl-bridge`` (add ``--features gpu`` for ``backend="gpu"``) and point
``STL_NATIVE_LIBRARY`` to ``target/release/libstl_bridge.so``; the module itself needs only NumPy
(and PyTorch for the ``*_device`` methods).

Arrays are ``float64`` / ``complex128``, maps are full-sky NESTED. A cotangent is
``dL/dRe + i dL/dIm``: the gradient returned by a ``vjp`` is that of
``Re sum(conj(cotangent) * values)``. Guides: :doc:`../user-guide/native-engine`,
:doc:`../user-guide/patch-engine`, :doc:`../user-guide/spectrum-engine`.

Engines
-------

.. currentmodule:: stl_port.native

.. autosummary::
   :toctree: generated
   :nosignatures:

   NativeEngine
   PatchEngine
   SpectrumEngine
   Snapshot

Optimiser and transforms
------------------------

.. autosummary::
   :toctree: generated

   native_lbfgs
   delouis_transform
   delouis_transform_vjp
