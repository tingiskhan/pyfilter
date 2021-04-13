from .parameter import ExtendedParameter


PRIOR_PREFIX = "prior__"


class PriorMixin(object):
    def register_prior(self, name, prior):
        prior_name = f"{PRIOR_PREFIX}{name}"
        self.add_module(prior_name, prior)
        self.register_parameter(name, ExtendedParameter(prior().sample(), requires_grad=False))

    def parameters_and_priors(self):
        for n, p in self.named_parameters():
            if "." not in n:
                yield p, self._modules[f"{PRIOR_PREFIX}{n}"]
            else:
                # TODO: Use recursion...
                sub_mod, name = n.split(".")
                yield p, self._modules[sub_mod]._modules[f"{PRIOR_PREFIX}{name}"]

    def priors(self):
        for _, m in self.parameters_and_priors():
            yield m
