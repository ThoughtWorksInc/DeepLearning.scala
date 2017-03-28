package org.typelevel.future.sde

import com.qifun.statelessFuture.Future
import com.thoughtworks.sde.core.{MonadicFactory, Preprocessor}
import macrocompat.bundle
import org.typelevel.future.scalaz.FutureInstances

import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.reflect.macros.whitebox
import scalaz.MonadError
import scala.language.experimental.macros
import scala.language.higherKinds

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@compileTimeOnly("enable macro paradise to expand macro annotations")
final class future extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro future.AnnotationBundle.macroTransform
}

object future extends MonadicFactory.WithTypeClass[({ type T[F[_]] = MonadError[F, Throwable] })#T, Future] {

  type MonadThrowable[F[_]] = MonadError[F, Throwable]

  override val typeClass: MonadThrowable[Future] = new FutureInstances[Unit]

  @bundle
  private[future] final class AnnotationBundle(context: whitebox.Context) extends Preprocessor(context) {

    import c.universe._

    def macroTransform(annottees: Tree*): Tree = {
      replaceDefBody(
        annottees, { body =>
          q"""
          org.typelevel.future.sde.future {
            import org.typelevel.future.sde.future.AutoImports._
            ${(new ComprehensionTransformer).transform(body)}
          }
        """
        }
      )
    }

  }

  @bundle
  private[future] class AwaitBundle(val c: whitebox.Context) {
    import c.universe._

    def prefixAwait(future: Tree): Tree = {
      val q"$methodName[$a]($future)" = c.macroApplication
      q"""
        _root_.com.thoughtworks.sde.core.MonadicFactory.Instructions.each[
          _root_.com.qifun.statelessFuture.Future.Stateless,
          $a
        ]($future)
      """
    }

    def postfixAwait: Tree = {
      val q"$ops.$methodName" = c.macroApplication
      val opsName = TermName(c.freshName("ops"))
      q"""
        val $opsName = $ops
        _root_.com.thoughtworks.sde.core.MonadicFactory.Instructions.each[
          _root_.com.qifun.statelessFuture.Future.Stateless,
          $opsName.A
        ]($opsName.underlying)
      """
    }

  }

  object AutoImports {

    import scala.language.implicitConversions

    implicit final class AwaitOps[A0](val underlying: Future[A0]) extends AnyVal {
      type A = A0
      def ! : A = macro AwaitBundle.postfixAwait
    }

    implicit def await[A](future: Future[A]): A = macro AwaitBundle.prefixAwait

  }
}
