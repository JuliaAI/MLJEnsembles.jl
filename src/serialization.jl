

MLJModelInterface.save(::MLJEnsembles.EitherEnsembleModel, fitresult) =
    MLJEnsembles.WrappedEnsemble(
        fitresult.atom,
        [save(fitresult.atom, fr) for fr in fitresult.ensemble]
    )

MLJModelInterface.restore(::MLJEnsembles.EitherEnsembleModel, fitresult) =
    MLJEnsembles.WrappedEnsemble(
        fitresult.atom,
        [restore(fitresult.atom, fr) for fr in fitresult.ensemble]
    )
