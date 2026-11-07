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
:doc:`../user-guide/patch-engine`, :doc:`../user-guide/spectrum-engine`,
:doc:`../user-guide/spin-statistics`, :doc:`../user-guide/native-unet`.

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

Network
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   NativeUNet

Optimiser and transforms
------------------------

.. autosummary::
   :toctree: generated

   native_lbfgs
   delouis_transform
   delouis_transform_vjp

Denoising helpers: ``stl_port.denoise``
---------------------------------------

The host-side parts of the denoising losses used by
``python/stl_ensemble/train_native_unet.py``: galactic masks, the binned half-mission spectrum
term ``cl`` over ``SpectrumEngine``, and the conditioning maps (needs healpy).

.. currentmodule:: stl_port.denoise

.. autosummary::
   :toctree: generated
   :nosignatures:

   galactic_mask
   stat_masks
   bin_edges
   BinnedSpectra
   SpectrumLoss
   conditioning
