## All instances of Layer Normalization in JoeyNMT

Our task is to write some 'missing documentation' for Layer Normalization in JoeyNMT. We should consider the following aspects: Where and how are pre- and post-norm implemented? What is the default behaviour? What are the specific differences between the two options and how do you control them?

### encoders.py

In `encoders.py` layer normalization is implemented on lines 215/216 and 252/253

```python
215:        self.layer_norm = (nn.LayerNorm(hidden_size, eps=1e-6) if kwargs.get(
216:            "layer_norm", "post") == "pre" else None)


252:        if self.layer_norm is not None:
253:            x = self.layer_norm(x)

```
Here, it is important to note that when the layers are built, the default value for layer_norm is pre, unlike in decoders.py. This means: When we instantiate this class not passing another value for layer normalization, we will automatically have the TransformerEncoder use layer normalization (nn.LayerNorm(hidden_size, eps=1e-6)). If we instantiate the class and do pass the value "post" for layer_norm, then the TransformerEncoder will not be using layer normalization - the attribute "self.layer_norm" will be equal to None, which is checked in the forward function before normalizing the desired layers (i.e. as per default). It is thus possible to change this value, but __I do have to check where__

### decoders.py
```python
534:        self.layer_norm = (nn.LayerNorm(hidden_size, eps=1e-6) if kwargs.get(
535:            "layer_norm", "post") == "pre" else None)

# in the forward() function
588:        if self.layer_norm is not None:
589:            x = self.layer_norm(x)

```
Here, it is important to note that when the layers are built, the default value for layer_norm is post, unlike in encoders.py. Additionally, the rest is very similarly built: When we instantiate this class not passing another value for layer normalization, we will automatically have the TransformerDecoder use layer normalization (nn.LayerNorm(hidden_size, eps=1e-6)). If we instantiate the class and do pass the value "pre" for layer_norm, then the TransformerDecoder will not be using layer normalization - the attribute "self.layer_norm" will be equal to None, which is checked in the forward function before normalizing the desired layers (i.e. as per default). It is thus possible to change this value, but __I do have to check where__

### transformer_layers.py
in transformer_layers.py, 

```python
141        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
```
As you can see in the source code, the layer normalization is initialised automatically in the class PositionwiseFeedForward. The default value is "post", but it accepts both "pre" and "post" as arguments, as can be seen in line 152. 

```python
151:      self._layer_norm_position = layer_norm
152:      assert self._layer_norm_position in {"pre", "post"}
```

In this forward function, the layer normalization happens either before or after the "forwarding action". It is thus not possible to not normalize when using this class, the question is just whether you normalize the layers before or after.

```python
154:      def forward(self, x: Tensor) -> Tensor:
155:              residual = x
156:              if self._layer_norm_position == "pre":
157:                  x = self.layer_norm(x)
158:
159:              x = self.pwff_layer(x) + self.alpha * residual
160:
161:              if self._layer_norm_position == "post":
162:                  x = self.layer_norm(x)
163:              return x
```

Now, where do we control whether it happens pre or post forwarding? This class is actually used in this file as well. For this, we need to elaborate more on what else happens there.

There are two further important classes: the TransformerEncoderLayer and the TransformerDecoderLayer.

The TransformerEncoderLayer has an attribute layer_norm with a default value "post" in line 222. It also automatically instantiates layer normalization.

```python
241:        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
```

Following this, in line 244, we see that the PositionwiseFeedForward class - from above - is called as self.feed_forward. It receives the layer_norm value that we have given TransformerEncoderLayer as input. Thus, for the encoder, we use "post"-normalization as a default in this specific case. We again have the same assert statement that we use in line 152 for the PositionwiseFeedForward class in line 258 in TransformerEncoderLayer class.

The forwarding function is also very similar to the one used in PositionwiseFeedForward. It is checked whether pre or post layer normalization is desired, and, once the layer input has been processed, before returning the modified tensor, it runs through the self.feed_forward attribute - which is essentially PositionwiseFeedForward used as an attribute. Thus, _we could even say that it normalizes twice: Once in the self.forward function, and once in the self.feed_forward function._ __is this statement correct?__

```python
260:     def forward(self, x: Tensor, mask: Tensor) -> Tensor:

271:        residual = x
272:        if self._layer_norm_position == "pre":
273:            x = self.layer_norm(x)
274:
275:        x, _ = self.src_src_att(x, x, x, mask)
276:        x = self.dropout(x) + self.alpha * residual
277:
278:        if self._layer_norm_position == "post":
279:            x = self.layer_norm(x)
280:
281:        out = self.feed_forward(x)
282:        return out
```

We also have the TransformerDecoderLayer class in this file. It is instantiated similarly: With a layer_norm default of "post" in line 299, a self.feed_forward based on the class PositionwiseFeedForward in line 323, and an assert statement in line 339 checking whether the indicated argument is "pre" or "post". 

Unlike above, we have two layer_norm attributes: self.x_layer_norm as self.dec_layer_norm.

```python
332:        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
333:        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
```

In the forward function, starting in line 341, we use the self.x_layer_norm either before or after the "target-target self-attention", and the self.dec_layer_norm either before or after the source-target cross-attention. At the very end, we again put the final tensor h2 in the self.feed_forward (that is, the PositionwiseFeedForward). Thus, _we could even say that it normalizes yet again the self.feed_forward function._ __is this statement correct?__

```python
398:        out = self.feed_forward(h2)
```

(Sidenote: Since the forward function contains many lines, we decided against including it in this file. You may find it here: https://github.com/lucaggett/joeynmt/blob/04d48e9112a216ec05a968332c85bc187e0c93c6/joeynmt/transformer_layers.py#L341)

