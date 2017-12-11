enablePlugins(Travis)

enablePlugins(SonatypeRelease)

enablePlugins(Optimization)

scalacOptions in ThisBuild ++= {
  import scala.math.Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers < Seq(2L, 12L)) {
    Seq("-Ybackend:GenBCode")
  } else {
    Nil
  }
}

lazy val secret = project settings(publishArtifact := false) configure { secret =>
  sys.env.get("GITHUB_PERSONAL_ACCESS_TOKEN") match {
    case Some(pat) =>
      import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider
      secret.addSbtFilesFromGit(
        "https://github.com/ThoughtWorksInc/tw-data-china-continuous-delivery-password.git",
        new UsernamePasswordCredentialsProvider(pat, ""),
        file("secret.sbt"))
    case None =>
      secret
  }
}
