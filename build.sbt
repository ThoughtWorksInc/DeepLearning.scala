parallelExecution in Global := false

lazy val Tape = project

lazy val tapefactories = project.dependsOn(Tape, logs)

includeFilter in unmanagedSources := (includeFilter in unmanagedSources).value && new SimpleFileFilter(_.isFile)

lazy val FloatHyperparameter = project.dependsOn(Hyperparameter, tapefactories, math, Loss)

lazy val INDArrayHyperparameter = project.dependsOn(DoubleHyperparameter)

val FloatRegex = """(?i:float)""".r

lazy val DoubleHyperparameter = project
  .dependsOn(Hyperparameter, tapefactories, math, Loss)
  .settings(
    sourceGenerators in Compile += Def.task {
      for {
        floatFile <- (unmanagedSources in Compile in FloatHyperparameter).value
        floatDirectory <- (unmanagedSourceDirectories in Compile in FloatHyperparameter).value
        relativeFile <- floatFile.relativeTo(floatDirectory)
      } yield {
        val floatSource = IO.read(floatFile, scala.io.Codec.UTF8.charSet)

        val doubleSource = FloatRegex.replaceAllIn(floatSource, { m =>
          m.matched match {
            case "Float" => "Double"
            case "float" => "double"
          }
        })

        val outputFile = (sourceManaged in Compile).value / relativeFile.getPath
        IO.write(outputFile, doubleSource, scala.io.Codec.UTF8.charSet)
        outputFile
      }
    }.taskValue
  )

lazy val Lift = project.dependsOn(Tape)

lazy val math = project.dependsOn(Lift)

lazy val Loss = project.dependsOn(Tape)
lazy val Hyperparameter = project.dependsOn(Tape)

lazy val logs = project

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(StandaloneUnidoc, TravisUnidocTitle)
  .settings(
    UnidocKeys.unidocProjectFilter in ScalaUnidoc in UnidocKeys.unidoc := {
      import Ordering.Implicits._
      if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
        inAggregates(LocalRootProject) -- inProjects(INDArrayHyperparameter)
      } else {
        inAggregates(LocalRootProject)
      }
    },
    addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3"),
    addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
    scalacOptions += "-Xexperimental"
  )

organization in ThisBuild := "com.thoughtworks.deeplearning"

crossScalaVersions := Seq("2.11.11", "2.12.2")

fork in ThisBuild in Test := true
